import Chip from '@mui/material/Chip';

export default function StatusPill({ label, active = false, tone = 'default' }) {
  return (
    <Chip
      size="small"
      label={label}
      className={`status-pill status-pill--${tone} ${active ? 'status-pill--active' : ''}`.trim()}
    />
  );
}
