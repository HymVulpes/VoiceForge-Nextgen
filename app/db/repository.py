"""
Database Repository Pattern
All queries centralized for debugging and caching
"""
from pathlib import Path
from typing import Optional, List
from datetime import datetime
import logging
from .models import VoiceModel, AudioConfig, ErrorLog, RuntimeMetric
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

class VoiceModelRepository:
    """Voice model CRUD operations"""
    
    @staticmethod
    def create(session: Session, name: str, pth_path: str, index_path: Optional[str] = None) -> VoiceModel:
        """Create new voice model with path validation"""
        pth_file = Path(pth_path)
        index_file = Path(index_path) if index_path else None
        
        model = VoiceModel(
            name=name,
            pth_path=str(pth_file.absolute()),
            index_path=str(index_file.absolute()) if index_file else None,
            file_exists=pth_file.exists(),
            file_size_mb=pth_file.stat().st_size / (1024*1024) if pth_file.exists() else None
        )
        
        session.add(model)
        session.commit()
        logger.info(f"Created voice model: {name} (exists={model.file_exists})")
        return model
    
    @staticmethod
    def get_by_id(session: Session, model_id: int) -> Optional[VoiceModel]:
        return session.query(VoiceModel).filter_by(id=model_id).first()
    
    @staticmethod
    def get_by_name(session: Session, name: str) -> Optional[VoiceModel]:
        return session.query(VoiceModel).filter_by(name=name).first()
    
    @staticmethod
    def get_all(session: Session) -> List[VoiceModel]:
        return session.query(VoiceModel).order_by(VoiceModel.last_used.desc()).all()
    
    @staticmethod
    def update_last_used(session: Session, model_id: int):
        model = session.query(VoiceModel).filter_by(id=model_id).first()
        if model:
            model.last_used = datetime.utcnow()
            session.commit()

class AudioConfigRepository:
    """Singleton audio configuration"""
    
    @staticmethod
    def get_or_create(session: Session) -> AudioConfig:
        """Get singleton config or create default"""
        config = session.query(AudioConfig).first()
        if not config:
            config = AudioConfig()
            session.add(config)
            session.commit()
            logger.info("Created default audio config")
        return config
    
    @staticmethod
    def update(session: Session, **kwargs) -> AudioConfig:
        """Update configuration"""
        config = AudioConfigRepository.get_or_create(session)
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
        session.commit()
        logger.info(f"Updated audio config: {kwargs}")
        return config

class ErrorLogRepository:
    """Error logging for debugging"""
    
    @staticmethod
    def log_error(
        session: Session,
        run_id: str,
        stage: str,
        error_type: str,
        error_message: str,
        file_origin: Optional[str] = None,
        stack_trace: Optional[str] = None,
        snapshot_path: Optional[str] = None
    ) -> ErrorLog:
        """Log error with full context"""
        error = ErrorLog(
            run_id=run_id,
            stage=stage,
            file_origin=file_origin,
            error_type=error_type,
            error_message=error_message,
            stack_trace=stack_trace,
            snapshot_path=snapshot_path
        )
        session.add(error)
        session.commit()
        logger.error(f"Logged error: {stage} - {error_type}")
        return error
    
    @staticmethod
    def get_by_run_id(session: Session, run_id: str) -> List[ErrorLog]:
        return session.query(ErrorLog).filter_by(run_id=run_id).order_by(ErrorLog.timestamp).all()
    
    @staticmethod
    def get_recent(session: Session, limit: int = 50) -> List[ErrorLog]:
        return session.query(ErrorLog).order_by(ErrorLog.timestamp.desc()).limit(limit).all()

class MetricsRepository:
    """Performance metrics logging"""
    
    @staticmethod
    def log_metric(
        session: Session,
        run_id: str,
        stage: str,
        latency_ms: float,
        **kwargs
    ) -> RuntimeMetric:
        """Log performance metric"""
        metric = RuntimeMetric(
            run_id=run_id,
            stage=stage,
            latency_ms=latency_ms,
            **kwargs
        )
        session.add(metric)
        session.commit()
        return metric
    
    @staticmethod
    def get_by_run_id(session: Session, run_id: str) -> List[RuntimeMetric]:
        return session.query(RuntimeMetric).filter_by(run_id=run_id).order_by(RuntimeMetric.timestamp).all()