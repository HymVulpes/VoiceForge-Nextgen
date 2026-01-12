"""
Database Models
SQLAlchemy ORM models for VoiceForge
"""
from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, Text
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

Base = declarative_base()

class VoiceModel(Base):
    """RVC voice model metadata"""
    __tablename__ = "voice_models"
    
    id = Column(Integer, primary_key=True)
    name = Column(String(255), nullable=False, unique=True)
    pth_path = Column(String(512), nullable=False)
    index_path = Column(String(512), nullable=True)
    file_exists = Column(Boolean, default=False)
    file_size_mb = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_used = Column(DateTime, default=datetime.utcnow)

class AudioConfig(Base):
    """Singleton audio configuration"""
    __tablename__ = "audio_config"
    
    id = Column(Integer, primary_key=True)
    input_device_index = Column(Integer, default=-1)
    input_device_name = Column(String(255), default="Unknown")
    virtual_output_index = Column(Integer, default=-1)
    output_device_name = Column(String(255), default="Unknown")
    sample_rate = Column(Integer, default=48000)
    buffer_size = Column(Integer, default=256)
    channels = Column(Integer, default=1)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class ErrorLog(Base):
    """Error logging for debugging"""
    __tablename__ = "error_logs"
    
    id = Column(Integer, primary_key=True)
    run_id = Column(String(36), nullable=False, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    stage = Column(String(50), nullable=False)
    file_origin = Column(String(255), nullable=True)
    error_type = Column(String(100), nullable=False)
    error_message = Column(Text, nullable=False)
    stack_trace = Column(Text, nullable=True)
    snapshot_path = Column(String(512), nullable=True)

class RuntimeMetric(Base):
    """Performance metrics"""
    __tablename__ = "runtime_metrics"
    
    id = Column(Integer, primary_key=True)
    run_id = Column(String(36), nullable=False, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    stage = Column(String(50), nullable=False)
    latency_ms = Column(Float, nullable=False)


