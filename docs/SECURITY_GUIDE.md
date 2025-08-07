# Sentiment Analyzer Pro - Security Guide

## Overview

This document outlines the comprehensive security measures implemented in Sentiment Analyzer Pro and provides guidance for maintaining a secure deployment in production environments.

## Security Architecture

### Defense in Depth
The system implements multiple layers of security:

1. **Input Validation Layer**
   - Text sanitization and validation
   - Length and content restrictions
   - Character encoding validation

2. **Application Security Layer**
   - Injection attack prevention
   - XSS protection
   - Security headers implementation

3. **Transport Security Layer**
   - TLS/SSL encryption
   - Certificate management
   - Secure communication protocols

4. **Infrastructure Security Layer**
   - Container security
   - Network isolation
   - Resource limitations

## Input Validation and Sanitization

### Text Validation Rules

**Length Restrictions**:
- Minimum text length: 1 character
- Maximum text length: 10,000 characters
- Automatic truncation for oversized inputs

**Content Filtering**:
```python
# Implemented in sentiment_analyzer_pro/utils/validation.py
class TextValidator:
    def validate_text(self, text: str) -> ValidationResult:
        # Check for minimum/maximum length
        # Validate character encoding
        # Filter inappropriate content
        # Check for potential security threats
```

**Character Encoding**:
- UTF-8 validation
- Control character filtering
- Non-printable character handling

### Injection Attack Prevention

**SQL Injection Protection**:
```python
# Detection patterns in sentiment_analyzer_pro/utils/security.py
SQL_INJECTION_PATTERNS = [
    r'(\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER)\b)',
    r'(\b(UNION|JOIN)\b.*\b(SELECT|FROM)\b)',
    r'(--|#|/\*|\*/)',
    r'(\b(OR|AND)\s+\w+\s*=\s*\w+)',
    r'(\'\s*(OR|AND)\s+\'?\w+\'?\s*=)',
]
```

**XSS Prevention**:
- HTML entity encoding
- Script tag detection and removal
- Event handler attribute filtering
- URL validation for embedded links

**Command Injection Protection**:
- Shell metacharacter filtering
- Command separator detection
- Path traversal prevention
- File extension validation

### Input Sanitization Process

1. **Text Normalization**:
   ```python
   def sanitize_input(self, text: str) -> str:
       # Remove/escape dangerous characters
       # Normalize whitespace
       # Validate encoding
       # Apply content filters
   ```

2. **Security Scanning**:
   - Pattern-based threat detection
   - Anomaly detection for unusual inputs
   - Rate-based attack detection

3. **Content Validation**:
   - Language detection
   - Content appropriateness checks
   - Spam/abuse detection

## Authentication and Authorization

### Current Implementation
- **Development**: No authentication (open access)
- **Production**: Recommended authentication methods below

### Recommended Authentication Methods

**API Key Authentication**:
```python
# Example implementation
@app.middleware("http")
async def authenticate_api_key(request: Request, call_next):
    api_key = request.headers.get("X-API-Key")
    if not validate_api_key(api_key):
        return JSONResponse(
            status_code=401,
            content={"error": "Invalid API key"}
        )
    response = await call_next(request)
    return response
```

**JWT Token Validation**:
```python
# Example JWT middleware
from jose import jwt

def validate_jwt_token(token: str) -> bool:
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        return True
    except jwt.JWTError:
        return False
```

## Rate Limiting and DDoS Protection

### Rate Limiting Configuration

**Nginx Rate Limiting**:
```nginx
# In nginx.conf
limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
limit_conn_zone $binary_remote_addr zone=addr:10m;

server {
    # Apply rate limiting
    limit_req zone=api burst=20 nodelay;
    limit_conn addr 10;
}
```

**Application-Level Rate Limiting**:
```python
# Implemented using slowapi/FastAPI
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

@app.post("/analyze")
@limiter.limit("10/minute")
async def analyze_sentiment(request: Request, text_input: TextInput):
    # Process request
```

### DDoS Protection Strategies

1. **Request Throttling**:
   - Per-IP rate limiting
   - Sliding window algorithms
   - Burst handling with queuing

2. **Resource Protection**:
   - Memory usage monitoring
   - CPU usage limits
   - Connection pooling

3. **Behavioral Analysis**:
   - Unusual request pattern detection
   - Automated blocking of suspicious IPs
   - Honeypot implementations

## Transport Security

### TLS/SSL Configuration

**SSL Certificate Management**:
```bash
# Generate self-signed certificate (development)
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
    -keyout ssl/key.pem \
    -out ssl/cert.pem \
    -subj "/C=US/ST=State/L=City/O=Organization/CN=localhost"

# For production, use certificates from trusted CA
```

**Nginx SSL Configuration**:
```nginx
server {
    listen 443 ssl http2;
    ssl_certificate /etc/nginx/ssl/cert.pem;
    ssl_certificate_key /etc/nginx/ssl/key.pem;
    
    # Security protocols
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES128-GCM-SHA256:ECDHE-RSA-AES256-GCM-SHA384;
    ssl_prefer_server_ciphers off;
    
    # Security headers
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains";
    add_header X-Frame-Options DENY;
    add_header X-Content-Type-Options nosniff;
    add_header X-XSS-Protection "1; mode=block";
}
```

## Container Security

### Docker Security Best Practices

**Dockerfile Security**:
```dockerfile
# Use specific version tags
FROM python:3.9-slim-bullseye

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Set proper permissions
COPY --chown=appuser:appuser . /app
USER appuser

# Minimize attack surface
RUN apt-get update && apt-get install -y --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*
```

**Container Runtime Security**:
```yaml
# docker-compose.production.yml security settings
services:
  sentiment-analyzer:
    security_opt:
      - no-new-privileges:true
    read_only: true
    tmpfs:
      - /tmp:noexec,nosuid,size=100m
    cap_drop:
      - ALL
    cap_add:
      - CHOWN
      - SETGID
      - SETUID
```

### Network Security

**Container Network Isolation**:
```yaml
networks:
  sentiment-network:
    driver: bridge
    ipam:
      config:
        - subnet: 172.20.0.0/16
```

**Firewall Configuration**:
```bash
# iptables rules for container security
iptables -A DOCKER-USER -i ext_if ! -s 172.20.0.0/16 -j DROP
iptables -A DOCKER-USER -i ext_if -m conntrack --ctstate RELATED,ESTABLISHED -j ACCEPT
```

## Monitoring and Alerting

### Security Event Monitoring

**Log Security Events**:
```python
import logging
security_logger = logging.getLogger('security')

def log_security_event(event_type: str, details: dict):
    security_logger.warning(
        f"Security event: {event_type}",
        extra={
            'event_type': event_type,
            'timestamp': datetime.utcnow(),
            'details': details
        }
    )
```

**Security Metrics**:
- Failed authentication attempts
- Rate limit violations
- Injection attempt detections
- Unusual request patterns
- Resource usage anomalies

### Alert Configuration

**Prometheus Alert Rules**:
```yaml
groups:
  - name: security_alerts
    rules:
      - alert: HighFailureRate
        expr: rate(sentiment_requests_failed[5m]) > 0.1
        labels:
          severity: warning
        annotations:
          summary: "High request failure rate detected"
      
      - alert: RateLimitViolations
        expr: rate(sentiment_rate_limit_violations[1m]) > 10
        labels:
          severity: critical
        annotations:
          summary: "Multiple rate limit violations detected"
```

## Vulnerability Management

### Regular Security Updates

**Dependency Scanning**:
```bash
# Use safety to check for known vulnerabilities
pip install safety
safety check

# Use bandit for security linting
pip install bandit
bandit -r sentiment_analyzer_pro/
```

**Container Image Scanning**:
```bash
# Use docker scan (powered by Snyk)
docker scan sentiment-analyzer-pro:latest

# Use Trivy for vulnerability scanning
trivy image sentiment-analyzer-pro:latest
```

### Security Testing

**Automated Security Tests**:
```python
# tests/test_security.py
def test_sql_injection_detection():
    malicious_input = "'; DROP TABLE users; --"
    with pytest.raises(ValidationError):
        validator.validate_input(malicious_input, "test")

def test_xss_prevention():
    xss_input = "<script>alert('XSS')</script>"
    sanitized = validator.validate_input(xss_input, "test")
    assert "<script>" not in sanitized
```

**Penetration Testing Checklist**:
- [ ] Input validation bypass attempts
- [ ] Authentication mechanism testing
- [ ] Rate limiting effectiveness
- [ ] SSL/TLS configuration validation
- [ ] Container escape attempts
- [ ] Network segmentation testing

## Data Protection

### Sensitive Data Handling

**Data Classification**:
- **Public**: API documentation, general information
- **Internal**: System logs, performance metrics
- **Confidential**: User input texts, analysis results
- **Restricted**: Authentication tokens, SSL keys

**Data Encryption**:
```python
# Example encryption for sensitive data
from cryptography.fernet import Fernet

def encrypt_sensitive_data(data: str, key: bytes) -> str:
    f = Fernet(key)
    encrypted_data = f.encrypt(data.encode())
    return encrypted_data.decode()
```

### Privacy Considerations

**Data Retention**:
- Input texts: Not stored permanently
- Analysis results: Cached temporarily (1 hour TTL)
- Logs: Rotated after 30 days
- Metrics: Aggregated, no personal data

**GDPR Compliance**:
- No personal data collection without consent
- Right to deletion implementation
- Data processing transparency
- Security breach notification procedures

## Incident Response

### Security Incident Handling

**Incident Response Plan**:
1. **Detection**: Automated monitoring and alerting
2. **Analysis**: Log analysis and threat assessment
3. **Containment**: Service isolation and traffic blocking
4. **Eradication**: Threat removal and system hardening
5. **Recovery**: Service restoration and monitoring
6. **Lessons Learned**: Post-incident analysis and improvements

**Emergency Procedures**:
```bash
# Emergency shutdown
docker-compose -f docker-compose.production.yml down

# Block malicious IP
iptables -A INPUT -s <malicious-ip> -j DROP

# Enable maintenance mode
touch /opt/sentiment-analyzer-pro/maintenance_mode
```

## Security Checklist

### Deployment Security Checklist

**Pre-Deployment**:
- [ ] Update all dependencies to latest secure versions
- [ ] Run security scans (bandit, safety, docker scan)
- [ ] Review and test security configurations
- [ ] Generate proper SSL certificates
- [ ] Configure firewall rules
- [ ] Set up monitoring and alerting

**Post-Deployment**:
- [ ] Verify SSL/TLS configuration
- [ ] Test rate limiting functionality
- [ ] Validate input validation mechanisms
- [ ] Check security headers
- [ ] Monitor security logs
- [ ] Perform basic penetration testing

### Ongoing Security Maintenance

**Daily**:
- [ ] Review security logs for anomalies
- [ ] Monitor performance metrics for unusual patterns
- [ ] Check system resource usage

**Weekly**:
- [ ] Review access logs
- [ ] Update security rules if needed
- [ ] Check for new security advisories

**Monthly**:
- [ ] Security dependency updates
- [ ] Certificate expiration checks
- [ ] Security configuration review
- [ ] Incident response plan testing

**Quarterly**:
- [ ] Full security audit
- [ ] Penetration testing
- [ ] Security training updates
- [ ] Disaster recovery testing

## Contact Information

**Security Team**: security@company.com  
**Emergency Contact**: +1-XXX-XXX-XXXX  
**Incident Reporting**: incidents@company.com

## Resources

- [OWASP Top 10](https://owasp.org/www-project-top-ten/)
- [CIS Controls](https://www.cisecurity.org/controls/)
- [NIST Cybersecurity Framework](https://www.nist.gov/cyberframework)
- [Docker Security Best Practices](https://docs.docker.com/engine/security/)
- [FastAPI Security](https://fastapi.tiangolo.com/tutorial/security/)