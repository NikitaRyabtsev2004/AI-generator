import { Area, AreaChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import { Typography } from '@mui/material';
import { formatDecimal } from '../../utils/text';

export default function TrainingChart({ history }) {
  if (!history?.length) {
    return (
      <div className="chart-empty">
        <Typography variant="body2">
          После запуска обучения здесь появится график loss по батчам.
        </Typography>
      </div>
    );
  }

  return (
    <div className="training-chart">
      <ResponsiveContainer width="100%" height={320}>
        <AreaChart data={history}>
          <defs>
            <linearGradient id="lossChartGradient" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#8ce1c7" stopOpacity={0.65} />
              <stop offset="95%" stopColor="#8ce1c7" stopOpacity={0} />
            </linearGradient>
          </defs>
          <CartesianGrid stroke="rgba(255,255,255,0.08)" vertical={false} />
          <XAxis
            dataKey="step"
            tick={{ fill: 'rgba(255,255,255,0.72)', fontSize: 12 }}
            axisLine={{ stroke: 'rgba(255,255,255,0.08)' }}
            tickLine={false}
          />
          <YAxis
            tick={{ fill: 'rgba(255,255,255,0.72)', fontSize: 12 }}
            axisLine={{ stroke: 'rgba(255,255,255,0.08)' }}
            tickLine={false}
            width={60}
          />
          <Tooltip
            contentStyle={{
              borderRadius: 18,
              border: '1px solid rgba(255,255,255,0.1)',
              background: 'rgba(8, 14, 24, 0.95)',
            }}
            labelFormatter={(value) => `Шаг ${value}`}
            formatter={(value, _name, payload) => [
              formatDecimal(value, 4),
              `loss • эпоха ${payload?.payload?.epoch}, батч ${payload?.payload?.batch}`,
            ]}
          />
          <Area
            type="monotone"
            dataKey="loss"
            stroke="#8ce1c7"
            strokeWidth={2.4}
            fill="url(#lossChartGradient)"
            dot={false}
          />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}
