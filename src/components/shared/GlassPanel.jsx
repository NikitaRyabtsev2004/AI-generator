import { Box } from '@mui/material';
import LiquidGlass from '../../LiquidGlass';
import '../../styles/glass-panel.css';

export default function GlassPanel({
  children,
  className = '',
  innerClassName = '',
  filterStyle
}) {
  return (
    <LiquidGlass
      width="100%"
      height="auto"
      className={`glass-panel ${className}`.trim()}
      filterStyle={filterStyle}
      style={{
        borderRadius: '24px',
        background: 'rgba(9, 18, 29, 0.22)',
      }}
    >
      <Box className={`glass-panel__inner ${innerClassName}`.trim()}>
        {children}
      </Box>
    </LiquidGlass>
  );
}
