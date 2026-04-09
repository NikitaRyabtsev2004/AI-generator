import React, { useState, useRef, forwardRef } from 'react';
const LiquidGlass = forwardRef(({
    width,
    height,
    baseFrequency = "0.007",
    numOctaves = 1,
    seed: initialSeed = 23,
    turbulenceType = "fractalNoise",
    blurDeviation = 15,
    displacementScale = 200,
    xChannelSelector = "R",
    yChannelSelector = "G",
    style = {},
    className = "",
    children,
    ...props
}, ref) => {
    const [seed] = useState(initialSeed);
    const wrapperRef = useRef(null);
    const combinedRef = (node) => {
        wrapperRef.current = node;
        if (typeof ref === 'function') ref(node);
        else if (ref) ref.current = node;
    };
    const uniqueId = `lg-${Math.random().toString(36).substr(2, 9)}`;
    return (
        <div
            ref={combinedRef}
            className={`liquid-glass ${className}`}
            style={{
                position: 'relative',
                width: typeof width === 'number' ? `${width}px` : width,
                height: typeof height === 'number' ? `${height}px` : height,
                borderRadius: '2px',
                overflow: 'hidden',
                isolation: 'isolate',
                background: 'rgba(255, 255, 255, 0.1)',
                boxShadow: '0 25px 50px rgba(0, 0, 0, 0.63), 0 0 0 1px rgba(255, 255, 255, 0.38), inset 0 1px 0 rgba(255, 255, 255, 0)',
                ...style
            }}
            {...props}
        >
            <svg
                style={{
                    position: 'absolute',
                    width: 0,
                    height: 0,
                    overflow: 'hidden',
                    pointerEvents: 'none'
                }}
            >
                <defs>
                    <filter id={`${uniqueId}-distortion`} x="-50%" y="-50%" width="200%" height="200%">
                        <feTurbulence
                            type={turbulenceType}
                            baseFrequency={baseFrequency}
                            numOctaves={numOctaves}
                            seed={seed}
                            result="turbulence"
                        />
                        <feGaussianBlur
                            in="turbulence"
                            stdDeviation={blurDeviation}
                            result="softMap"
                        />
                        <feComponentTransfer in="softMap" result="maskedMap">
                            <feFuncA type="linear" slope="0" intercept="1" />
                        </feComponentTransfer>
                        <feDisplacementMap
                            in="SourceGraphic"
                            in2="softMap"
                            scale={displacementScale}
                            xChannelSelector={xChannelSelector}
                            yChannelSelector={yChannelSelector}
                        />
                    </filter>
                </defs>
            </svg>
            <div
                style={{
                    position: 'absolute', inset: 0, zIndex: 0
                }}
            />
            <div
                style={{
                    position: 'absolute',
                    width: '100%', height: '100%',
                    backdropFilter: 'blur(3px)',
                    filter: `url(#${uniqueId}-distortion)`,
                    opacity: 1,
                    background: 'radial-gradient(circle at 50% 50%, rgba(255, 255, 255, 0.198) 0%, transparent 50%)',
                    zIndex: 1,
                    pointerEvents: 'none'
                }}
            />
            <div
                style={{
                    position: 'relative',
                    zIndex: 10,
                    width: '100%',
                    height: '100%',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center'
                }}
            >
                {children}
            </div>
        </div>
    );
});
LiquidGlass.displayName = 'LiquidGlass';
export default LiquidGlass;