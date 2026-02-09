"""
Database module for Form Mapping System.
Uses PostgreSQL with SQLAlchemy ORM.
All tables prefixed with 'od1_' to avoid conflicts in shared database.
"""

import os
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path

from sqlalchemy import (
    create_engine, Column, String, Integer, Boolean, Text, DateTime,
    ForeignKey, JSON, Float, Index, UniqueConstraint, text
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship, Session, joinedload, selectinload
from dotenv import load_dotenv

# Load environment variables
load_dotenv(Path(__file__).parent / ".env")

# Logger setup
logger = logging.getLogger("database")
if not logger.handlers:
    logger.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s")
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

# Database URL handling
DATABASE_URL = os.environ.get("DATABASE_URL")
if DATABASE_URL and DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

# SQLAlchemy setup
Base = declarative_base()

# Engine and session (lazy initialization)
_engine = None
_SessionLocal = None


def get_engine():
    """Get or create SQLAlchemy engine."""
    global _engine
    if _engine is None:
        if not DATABASE_URL:
            raise ValueError("DATABASE_URL environment variable not set")
        _engine = create_engine(
            DATABASE_URL,
            pool_pre_ping=True,
            pool_size=5,
            max_overflow=10
        )
    return _engine


def get_session() -> Session:
    """Get a new database session."""
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(bind=get_engine())
    return _SessionLocal()


# ============================================================================
# Models - All prefixed with od1_
# ============================================================================

class Od1Form(Base):
    """Form metadata record."""
    __tablename__ = "od1_forms"

    id = Column(Integer, primary_key=True, autoincrement=True)
    opportunity_id = Column(String(255), nullable=False, index=True)
    pdf_checksum = Column(String(64), nullable=False)  # SHA-256
    pdf_filename = Column(String(512))
    pdf_filepath = Column(String(1024))  # Local path (deprecated, kept for backward compat)
    pdf_s3_key = Column(String(512))  # S3 key like "od1_/pdfs/123-abc.pdf"
    process_date = Column(DateTime, default=datetime.utcnow)
    is_form = Column(Boolean, default=True)
    form_description = Column(Text)
    general_info_needed = Column(JSON, default=list)  # List of strings
    from_cache = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    questions = relationship("Od1Question", back_populates="form", cascade="all, delete-orphan")
    pdf_fields = relationship("Od1PdfField", back_populates="form", cascade="all, delete-orphan")
    field_mappings = relationship("Od1FieldMapping", back_populates="form", cascade="all, delete-orphan")

    # Unique constraint on opportunity_id + checksum
    __table_args__ = (
        UniqueConstraint("opportunity_id", "pdf_checksum", name="uq_od1_form_opp_checksum"),
        Index("ix_od1_forms_checksum", "pdf_checksum"),
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary matching JSONL format."""
        return {
            "opportunity_id": self.opportunity_id,
            "pdf_checksum": self.pdf_checksum,
            "pdf_filename": self.pdf_filename,
            "pdf_filepath": self.pdf_filepath,
            "pdf_s3_key": self.pdf_s3_key,
            "process_date": self.process_date.isoformat() if self.process_date else None,
            "from_cache": self.from_cache,
            "form_structure": {
                "is_form": self.is_form,
                "form_description": self.form_description,
                "general_info_needed": self.general_info_needed or [],
                "questions": [q.to_dict() for q in self.questions],
                "pdf_form_fields": [f.to_dict() for f in self.pdf_fields] if self.pdf_fields else None,
                "field_mappings": [m.to_dict() for m in self.field_mappings] if self.field_mappings else None,
            }
        }


class Od1Question(Base):
    """Question within a form."""
    __tablename__ = "od1_questions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    form_id = Column(Integer, ForeignKey("od1_forms.id", ondelete="CASCADE"), nullable=False)
    question_id = Column(String(50))  # e.g., 'fq_0', 'fq_1'
    question_number = Column(String(50))
    question_text_short = Column(Text)
    question_text = Column(Text)
    question_type = Column(String(100))
    char_limit = Column(Integer)
    is_required = Column(Boolean, default=False)
    is_agreement = Column(Boolean, default=False)
    sub_question_of = Column(Text)  # Can be long question references
    unique_structure = Column(Boolean, default=False)
    mapped_pdf_fields = Column(JSON, default=list)  # List of field names
    sort_order = Column(Integer, default=0)

    # Relationships
    form = relationship("Od1Form", back_populates="questions")
    mappings = relationship("Od1QuestionMapping", back_populates="question", cascade="all, delete-orphan")

    __table_args__ = (
        Index("ix_od1_questions_form_id", "form_id"),
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary matching JSONL format."""
        return {
            "question_id": self.question_id,
            "question_number": self.question_number,
            "question_text_short": self.question_text_short,
            "question_text": self.question_text,
            "question_type": self.question_type,
            "char_limit": self.char_limit,
            "is_required": self.is_required,
            "is_agreement": self.is_agreement,
            "sub_question_of": self.sub_question_of,
            "unique_structure": self.unique_structure,
            "mapped_pdf_fields": self.mapped_pdf_fields or [],
            "mappings": [m.to_dict() for m in self.mappings]
        }


class Od1QuestionMapping(Base):
    """Mapping from a question to a canonical question."""
    __tablename__ = "od1_question_mappings"

    id = Column(Integer, primary_key=True, autoincrement=True)
    question_id = Column(Integer, ForeignKey("od1_questions.id", ondelete="CASCADE"), nullable=False)
    canonical_version_id = Column(String(64), nullable=False)
    canonical_question_id = Column(String(50))  # e.g., 'cq_001'
    similarity_score = Column(Integer)  # 1-5

    # Relationships
    question = relationship("Od1Question", back_populates="mappings")

    __table_args__ = (
        Index("ix_od1_question_mappings_question_id", "question_id"),
        Index("ix_od1_question_mappings_version", "canonical_version_id"),
        UniqueConstraint("question_id", "canonical_version_id", name="uq_od1_qmap_q_version"),
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "canonical_version_id": self.canonical_version_id,
            "canonical_question_id": self.canonical_question_id,
            "similarity_score": self.similarity_score
        }


class Od1PdfField(Base):
    """PDF form field extracted from document."""
    __tablename__ = "od1_pdf_fields"

    id = Column(Integer, primary_key=True, autoincrement=True)
    form_id = Column(Integer, ForeignKey("od1_forms.id", ondelete="CASCADE"), nullable=False)
    field_name = Column(String(512), nullable=False)
    field_type = Column(String(50))  # text, checkbox, radio, dropdown, signature
    options = Column(JSON, default=list)  # For dropdown/radio options
    rect = Column(JSON)  # [x1, y1, x2, y2]
    page_number = Column(Integer)

    # Relationships
    form = relationship("Od1Form", back_populates="pdf_fields")

    __table_args__ = (
        Index("ix_od1_pdf_fields_form_id", "form_id"),
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "field_name": self.field_name,
            "field_type": self.field_type,
            "options": self.options or [],
            "rect": self.rect,
            "page_number": self.page_number
        }


class Od1FieldMapping(Base):
    """Mapping between PDF field and question."""
    __tablename__ = "od1_field_mappings"

    id = Column(Integer, primary_key=True, autoincrement=True)
    form_id = Column(Integer, ForeignKey("od1_forms.id", ondelete="CASCADE"), nullable=False)
    pdf_field_name = Column(String(512), nullable=False)
    pdf_field_type = Column(String(50))
    question_number = Column(String(50))
    question_text_short = Column(Text)
    confidence = Column(String(20))  # high, medium, low
    mapping_notes = Column(Text)

    # Relationships
    form = relationship("Od1Form", back_populates="field_mappings")

    __table_args__ = (
        Index("ix_od1_field_mappings_form_id", "form_id"),
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "pdf_field_name": self.pdf_field_name,
            "pdf_field_type": self.pdf_field_type,
            "question_number": self.question_number,
            "question_text_short": self.question_text_short,
            "confidence": self.confidence,
            "mapping_notes": self.mapping_notes
        }


class Od1CanonicalVersion(Base):
    """Canonical questions version."""
    __tablename__ = "od1_canonical_versions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    version_id = Column(String(64), unique=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    model_used = Column(String(100))
    input_token_count = Column(Integer)
    output_token_count = Column(Integer)
    processing_notes = Column(Text)
    is_latest = Column(Boolean, default=False)

    # Relationships
    questions = relationship("Od1CanonicalQuestion", back_populates="version", cascade="all, delete-orphan")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "version_id": self.version_id,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "model_used": self.model_used,
            "input_token_count": self.input_token_count,
            "output_token_count": self.output_token_count,
            "processing_notes": self.processing_notes,
            "is_latest": self.is_latest,
            "canonical_questions": [q.to_dict() for q in self.questions]
        }


class Od1CanonicalQuestion(Base):
    """Canonical question definition."""
    __tablename__ = "od1_canonical_questions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    version_id = Column(Integer, ForeignKey("od1_canonical_versions.id", ondelete="CASCADE"), nullable=False)
    canonical_question_id = Column(String(50), nullable=False)  # e.g., 'cq_001'
    question_text = Column(Text, nullable=False)
    question_type = Column(String(100))
    char_limit = Column(Integer)
    is_agreement = Column(Boolean, default=False)
    sub_question_of = Column(Text)  # Can be long question references
    unique_structure = Column(Boolean, default=False)

    # Relationships
    version = relationship("Od1CanonicalVersion", back_populates="questions")

    __table_args__ = (
        Index("ix_od1_canonical_questions_version_id", "version_id"),
        UniqueConstraint("version_id", "canonical_question_id", name="uq_od1_cq_version_qid"),
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "canonical_question_id": self.canonical_question_id,
            "question_text": self.question_text,
            "question_type": self.question_type,
            "char_limit": self.char_limit,
            "is_agreement": self.is_agreement,
            "sub_question_of": self.sub_question_of,
            "unique_structure": self.unique_structure
        }


class Od1SimilarityRating(Base):
    """Human rating of LLM-assigned similarity scores."""
    __tablename__ = "od1_similarity_ratings"

    id = Column(Integer, primary_key=True, autoincrement=True)
    canonical_version_id = Column(String(64), nullable=False, index=True)
    form_opportunity_id = Column(String(255), nullable=False)
    form_question_number = Column(String(50), nullable=False)
    canonical_question_id = Column(String(50), nullable=False)
    llm_similarity_score = Column(Integer, nullable=False)  # 1-5
    is_same = Column(Boolean, nullable=False)
    rated_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index("ix_od1_ratings_version_score", "canonical_version_id", "llm_similarity_score"),
    )


# ============================================================================
# Database Operations
# ============================================================================

def init_db():
    """Initialize database tables."""
    engine = get_engine()
    Base.metadata.create_all(engine)
    logger.info("Database tables created/verified")


def drop_all_tables():
    """Drop all od1_ tables. USE WITH CAUTION."""
    engine = get_engine()
    Base.metadata.drop_all(engine)
    logger.warning("All od1_ tables dropped")


# ============================================================================
# Form Record Operations (compatible with JSONL interface)
# ============================================================================

def load_form_records_db() -> List[Dict[str, Any]]:
    """
    Load all form records from database.
    Returns list of dictionaries in same format as JSONL.
    Uses eager loading to avoid N+1 query problem.
    """
    session = get_session()
    try:
        forms = session.query(Od1Form).options(
            selectinload(Od1Form.questions).selectinload(Od1Question.mappings),
            selectinload(Od1Form.pdf_fields),
            selectinload(Od1Form.field_mappings)
        ).all()
        records = [form.to_dict() for form in forms]
        logger.info(f"Loaded {len(records)} form records from database")
        return records
    finally:
        session.close()


def save_form_record_db(record: Dict[str, Any]) -> Od1Form:
    """
    Save a form record to database.
    Updates if exists (same opp_id + checksum), inserts if new.
    """
    session = get_session()
    try:
        opp_id = record.get("opportunity_id")
        checksum = record.get("pdf_checksum")
        form_structure = record.get("form_structure", {})

        # Check if exists
        existing = session.query(Od1Form).filter_by(
            opportunity_id=opp_id,
            pdf_checksum=checksum
        ).first()

        if existing:
            form = existing
            # Update fields
            form.pdf_filename = record.get("pdf_filename")
            form.pdf_filepath = record.get("pdf_filepath")
            form.pdf_s3_key = record.get("pdf_s3_key")
            form.is_form = form_structure.get("is_form", True)
            form.form_description = form_structure.get("form_description")
            form.general_info_needed = form_structure.get("general_info_needed", [])
            form.from_cache = record.get("from_cache", False)
            form.updated_at = datetime.utcnow()

            # Clear existing related records
            session.query(Od1Question).filter_by(form_id=form.id).delete()
            session.query(Od1PdfField).filter_by(form_id=form.id).delete()
            session.query(Od1FieldMapping).filter_by(form_id=form.id).delete()
        else:
            # Create new form
            process_date = record.get("process_date")
            if isinstance(process_date, str):
                try:
                    process_date = datetime.fromisoformat(process_date.replace("Z", "+00:00"))
                except:
                    process_date = datetime.utcnow()

            form = Od1Form(
                opportunity_id=opp_id,
                pdf_checksum=checksum,
                pdf_filename=record.get("pdf_filename"),
                pdf_filepath=record.get("pdf_filepath"),
                pdf_s3_key=record.get("pdf_s3_key"),
                process_date=process_date,
                is_form=form_structure.get("is_form", True),
                form_description=form_structure.get("form_description"),
                general_info_needed=form_structure.get("general_info_needed", []),
                from_cache=record.get("from_cache", False)
            )
            session.add(form)
            session.flush()  # Get form.id

        # Add questions
        questions = form_structure.get("questions", [])
        for idx, q_data in enumerate(questions):
            question = Od1Question(
                form_id=form.id,
                question_id=q_data.get("question_id") or f"fq_{idx}",
                question_number=q_data.get("question_number"),
                question_text_short=q_data.get("question_text_short"),
                question_text=q_data.get("question_text"),
                question_type=q_data.get("question_type"),
                char_limit=q_data.get("char_limit"),
                is_required=q_data.get("is_required", False),
                is_agreement=q_data.get("is_agreement", False),
                sub_question_of=q_data.get("sub_question_of"),
                unique_structure=q_data.get("unique_structure", False),
                mapped_pdf_fields=q_data.get("mapped_pdf_fields", []),
                sort_order=idx
            )
            session.add(question)
            session.flush()

            # Add question mappings
            for m_data in q_data.get("mappings", []):
                mapping = Od1QuestionMapping(
                    question_id=question.id,
                    canonical_version_id=m_data.get("canonical_version_id"),
                    canonical_question_id=m_data.get("canonical_question_id"),
                    similarity_score=m_data.get("similarity_score")
                )
                session.add(mapping)

        # Add PDF fields
        pdf_fields = form_structure.get("pdf_form_fields") or []
        for f_data in pdf_fields:
            pdf_field = Od1PdfField(
                form_id=form.id,
                field_name=f_data.get("field_name"),
                field_type=f_data.get("field_type"),
                options=f_data.get("options", []),
                rect=f_data.get("rect"),
                page_number=f_data.get("page_number")
            )
            session.add(pdf_field)

        # Add field mappings
        field_mappings = form_structure.get("field_mappings") or []
        for fm_data in field_mappings:
            field_mapping = Od1FieldMapping(
                form_id=form.id,
                pdf_field_name=fm_data.get("pdf_field_name"),
                pdf_field_type=fm_data.get("pdf_field_type"),
                question_number=fm_data.get("question_number"),
                question_text_short=fm_data.get("question_text_short"),
                confidence=fm_data.get("confidence"),
                mapping_notes=fm_data.get("mapping_notes")
            )
            session.add(field_mapping)

        session.commit()
        logger.info(f"Saved form record: {opp_id} ({checksum[:12]}...)")
        return form

    except Exception as e:
        session.rollback()
        logger.error(f"Failed to save form record: {e}")
        raise
    finally:
        session.close()


def save_all_form_records_db(records: List[Dict[str, Any]]) -> int:
    """Save all form records to database."""
    count = 0
    for record in records:
        try:
            save_form_record_db(record)
            count += 1
        except Exception as e:
            logger.error(f"Failed to save record {record.get('opportunity_id')}: {e}")
    return count


def get_form_by_opp_and_checksum(opp_id: str, checksum: str) -> Optional[Dict[str, Any]]:
    """Get a form record by opportunity ID and checksum."""
    session = get_session()
    try:
        form = session.query(Od1Form).filter_by(
            opportunity_id=opp_id,
            pdf_checksum=checksum
        ).first()
        return form.to_dict() if form else None
    finally:
        session.close()


# ============================================================================
# Canonical Version Operations
# ============================================================================

def save_canonical_version_db(version_data: Dict[str, Any]) -> Od1CanonicalVersion:
    """Save a canonical version to database."""
    session = get_session()
    try:
        version_id = version_data.get("version_id")

        # Check if exists
        existing = session.query(Od1CanonicalVersion).filter_by(version_id=version_id).first()
        if existing:
            logger.info(f"Canonical version {version_id} already exists, skipping")
            return existing

        # Create version
        created_at = version_data.get("created_at")
        if isinstance(created_at, str):
            try:
                created_at = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
            except:
                created_at = datetime.utcnow()

        version = Od1CanonicalVersion(
            version_id=version_id,
            created_at=created_at,
            model_used=version_data.get("model_used"),
            input_token_count=version_data.get("input_token_count"),
            output_token_count=version_data.get("output_token_count"),
            processing_notes=version_data.get("processing_notes"),
            is_latest=version_data.get("is_latest", False)
        )
        session.add(version)
        session.flush()

        # Add canonical questions
        for cq_data in version_data.get("canonical_questions", []):
            cq = Od1CanonicalQuestion(
                version_id=version.id,
                canonical_question_id=cq_data.get("canonical_question_id"),
                question_text=cq_data.get("question_text"),
                question_type=cq_data.get("question_type"),
                char_limit=cq_data.get("char_limit"),
                is_agreement=cq_data.get("is_agreement", False),
                sub_question_of=cq_data.get("sub_question_of"),
                unique_structure=cq_data.get("unique_structure", False)
            )
            session.add(cq)

        session.commit()
        logger.info(f"Saved canonical version: {version_id} with {len(version_data.get('canonical_questions', []))} questions")
        return version

    except Exception as e:
        session.rollback()
        logger.error(f"Failed to save canonical version: {e}")
        raise
    finally:
        session.close()


def list_canonical_versions_db() -> List[Dict[str, Any]]:
    """List all canonical versions with eager loading."""
    session = get_session()
    try:
        versions = session.query(Od1CanonicalVersion).options(
            selectinload(Od1CanonicalVersion.questions)
        ).order_by(Od1CanonicalVersion.created_at.desc()).all()
        return [v.to_dict() for v in versions]
    finally:
        session.close()


def get_canonical_version_db(version_id: str) -> Optional[Dict[str, Any]]:
    """Get a canonical version by ID with eager loading."""
    session = get_session()
    try:
        version = session.query(Od1CanonicalVersion).options(
            selectinload(Od1CanonicalVersion.questions)
        ).filter_by(version_id=version_id).first()
        return version.to_dict() if version else None
    finally:
        session.close()


def set_latest_canonical_version_db(version_id: str) -> bool:
    """Set a version as the latest."""
    session = get_session()
    try:
        # Clear existing latest
        session.query(Od1CanonicalVersion).update({"is_latest": False})

        # Set new latest
        result = session.query(Od1CanonicalVersion).filter_by(version_id=version_id).update({"is_latest": True})
        session.commit()
        return result > 0
    except Exception as e:
        session.rollback()
        logger.error(f"Failed to set latest version: {e}")
        return False
    finally:
        session.close()


# ============================================================================
# Similarity Rating Operations
# ============================================================================

def save_similarity_rating_db(
    canonical_version_id: str,
    form_opportunity_id: str,
    form_question_number: str,
    canonical_question_id: str,
    llm_similarity_score: int,
    is_same: bool
) -> Od1SimilarityRating:
    """Save a single similarity rating."""
    session = get_session()
    try:
        rating = Od1SimilarityRating(
            canonical_version_id=canonical_version_id,
            form_opportunity_id=form_opportunity_id,
            form_question_number=form_question_number,
            canonical_question_id=canonical_question_id,
            llm_similarity_score=llm_similarity_score,
            is_same=is_same
        )
        session.add(rating)
        session.commit()
        logger.info(f"Saved rating: score={llm_similarity_score}, is_same={is_same}")
        return rating
    except Exception as e:
        session.rollback()
        logger.error(f"Failed to save rating: {e}")
        raise
    finally:
        session.close()


def get_rating_counts_by_score_db(version_id: str) -> Dict[int, int]:
    """
    Get rating counts grouped by LLM similarity score.
    Returns dict like {1: 5, 2: 10, 3: 15, 4: 8, 5: 3}
    """
    session = get_session()
    try:
        from sqlalchemy import func
        results = session.query(
            Od1SimilarityRating.llm_similarity_score,
            func.count(Od1SimilarityRating.id)
        ).filter(
            Od1SimilarityRating.canonical_version_id == version_id
        ).group_by(
            Od1SimilarityRating.llm_similarity_score
        ).all()

        counts = {score: 0 for score in range(1, 6)}
        for score, count in results:
            counts[score] = count
        return counts
    finally:
        session.close()


def get_ratings_by_score_db(version_id: str, score: int) -> List[bool]:
    """
    Get all is_same values for a specific LLM score.
    Returns list of booleans for bootstrap resampling.
    """
    session = get_session()
    try:
        results = session.query(Od1SimilarityRating.is_same).filter(
            Od1SimilarityRating.canonical_version_id == version_id,
            Od1SimilarityRating.llm_similarity_score == score
        ).all()
        return [r[0] for r in results]
    finally:
        session.close()


def get_all_ratings_db(version_id: str) -> Dict[int, List[bool]]:
    """
    Get all is_same values grouped by LLM score.
    Returns dict like {1: [True, False, ...], 2: [...], ...}
    """
    session = get_session()
    try:
        results = session.query(
            Od1SimilarityRating.llm_similarity_score,
            Od1SimilarityRating.is_same
        ).filter(
            Od1SimilarityRating.canonical_version_id == version_id
        ).all()

        ratings = {score: [] for score in range(1, 6)}
        for score, is_same in results:
            ratings[score].append(is_same)
        return ratings
    finally:
        session.close()


# ============================================================================
# Utility Functions
# ============================================================================

def test_connection() -> bool:
    """Test database connection."""
    try:
        engine = get_engine()
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        logger.info("Database connection successful")
        return True
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        return False


if __name__ == "__main__":
    # Test connection
    if test_connection():
        print("Database connection successful!")
        init_db()
        print("Tables created/verified!")
    else:
        print("Database connection failed!")
