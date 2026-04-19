import {Box} from '@mui/material';
import LiquidGlass from '../../LiquidGlass';
import '../../styles/glass-panel.css';

export default function GlassPanel({
                                       children,
                                       className = '',
                                       innerClassName = '',
                                       style = {},
                                       filterStyle,
                                       innerSx = {},
                                       innerStyle = {},
                                   }) {
    return (
        <LiquidGlass
            width="100%"
            height="auto"
            className={`glass-panel ${className}`.trim()}
            filterStyle={filterStyle}
            style={{
                borderRadius: '16px',
                background: 'rgb(0 0 0 / 0.51)',
                ...style,
            }}
        >
            <Box
                className={`glass-panel__inner ${innerClassName}`.trim()}
                sx={innerSx}
                style={innerStyle}
            >
                {children}
            </Box>
        </LiquidGlass>
    );
}