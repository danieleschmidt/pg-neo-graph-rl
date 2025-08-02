"""Monitoring and observability package for PG-Neo-Graph-RL."""

from .metrics import (
    MetricsCollector,
    get_metrics_collector,
    initialize_metrics,
    record_training_loss,
    record_training_accuracy,
    set_active_agents,
    record_communication_latency,
    set_graph_size,
    timer_context,
    time_jax_compilation,
    time_graph_processing
)

from .logging_config import (
    setup_logging,
    setup_logging_from_env,
    get_logger,
    LogContext,
    log_function_call,
    log_execution_time,
    get_training_logger,
    get_federated_logger,
    get_graph_logger,
    get_monitoring_logger,
    JSONFormatter,
    ColoredFormatter
)

from .tracing import (
    Tracer,
    Span,
    get_tracer,
    initialize_tracing,
    start_span,
    get_current_span,
    span,
    trace_function,
    trace_class_methods,
    generate_correlation_id,
    get_correlation_id,
    set_correlation_id,
    inject_trace_headers,
    extract_trace_headers,
    record_performance_metric,
    get_performance_summary
)

from .health import (
    HealthStatus,
    HealthCheck,
    AsyncHealthCheck,
    HealthCheckResult,
    HealthManager,
    get_health_manager,
    initialize_health_checks,
    register_health_check,
    run_health_check,
    get_health_status,
    liveness_check,
    readiness_check,
    health_check_endpoint,
    JAXHealthCheck,
    MemoryHealthCheck,
    DiskSpaceHealthCheck,
    MetricsHealthCheck
)

__all__ = [
    # Metrics
    'MetricsCollector',
    'get_metrics_collector',
    'initialize_metrics',
    'record_training_loss',
    'record_training_accuracy',
    'set_active_agents',
    'record_communication_latency',
    'set_graph_size',
    'timer_context',
    'time_jax_compilation',
    'time_graph_processing',
    
    # Logging
    'setup_logging',
    'setup_logging_from_env',
    'get_logger',
    'LogContext',
    'log_function_call',
    'log_execution_time',
    'get_training_logger',
    'get_federated_logger',
    'get_graph_logger',
    'get_monitoring_logger',
    'JSONFormatter',
    'ColoredFormatter',
    
    # Tracing
    'Tracer',
    'Span',
    'get_tracer',
    'initialize_tracing',
    'start_span',
    'get_current_span',
    'span',
    'trace_function',
    'trace_class_methods',
    'generate_correlation_id',
    'get_correlation_id',
    'set_correlation_id',
    'inject_trace_headers',
    'extract_trace_headers',
    'record_performance_metric',
    'get_performance_summary',
    
    # Health checks
    'HealthStatus',
    'HealthCheck',
    'AsyncHealthCheck',
    'HealthCheckResult',
    'HealthManager',
    'get_health_manager',
    'initialize_health_checks',
    'register_health_check',
    'run_health_check',
    'get_health_status',
    'liveness_check',
    'readiness_check',
    'health_check_endpoint',
    'JAXHealthCheck',
    'MemoryHealthCheck',
    'DiskSpaceHealthCheck',
    'MetricsHealthCheck'
]


def initialize_monitoring(
    metrics_enabled: bool = True,
    prometheus_port: int = 8000,
    logging_level: str = "INFO",
    logging_format: str = "text",
    tracing_enabled: bool = True,
    health_checks_enabled: bool = True
):
    """Initialize all monitoring components.
    
    Args:
        metrics_enabled: Enable metrics collection
        prometheus_port: Port for Prometheus metrics server
        logging_level: Logging level
        logging_format: Logging format ('text' or 'json')
        tracing_enabled: Enable distributed tracing
        health_checks_enabled: Enable health checks
    
    Returns:
        Dictionary with initialized components
    """
    components = {}
    
    # Initialize logging
    logger = setup_logging(level=logging_level, format_type=logging_format)
    components['logger'] = logger
    logger.info("Logging initialized")
    
    # Initialize metrics
    if metrics_enabled:
        metrics_collector = initialize_metrics(
            enable_prometheus=True, 
            prometheus_port=prometheus_port
        )
        components['metrics'] = metrics_collector
        logger.info("Metrics collection initialized")
        
        try:
            metrics_collector.start_prometheus_server()
            logger.info(f"Prometheus server started on port {prometheus_port}")
        except Exception as e:
            logger.warning(f"Failed to start Prometheus server: {e}")
    
    # Initialize tracing
    if tracing_enabled:
        tracer = initialize_tracing()
        components['tracer'] = tracer
        logger.info("Distributed tracing initialized")
    
    # Initialize health checks
    if health_checks_enabled:
        health_manager = initialize_health_checks()
        components['health'] = health_manager
        logger.info("Health checks initialized")
    
    logger.info("All monitoring components initialized successfully")
    return components