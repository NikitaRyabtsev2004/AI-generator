import { Typography } from '@mui/material';

export default function MetricCard({ label, value, hint }) {
  return (
    <div className="metric-card">
      <Typography variant="caption" className="metric-card__label">
        {label}
      </Typography>
      <Typography variant="h4" className="metric-card__value">
        {value}
      </Typography>
      <Typography variant="caption" className="metric-card__hint">
        {hint}
      </Typography>
    </div>
  );
}
