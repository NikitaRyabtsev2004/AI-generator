import { Area, AreaChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import { Typography } from '@mui/material';
import { formatDecimal } from '../../utils/text';

export default function TrainingChart({ history }) {
  if (!history?.length) {
    return (
      <div className="chart-empty">
        <Typography variant="body2">
          После запуска обучения здесь появится график `loss` по батчам.
        </Typography>
      </div>
    );
  }

  return (
    <div className="training-chart">
      <div className="training-chart__canvas">
        <ResponsiveContainer width="100%" height="100%">
        <AreaChart
          data={history}
          margin={{ top: 12, right: 14, left: 2, bottom: 6 }}
        >
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
            domain={[
              (dataMin) => {
                const min = Number(dataMin) || 0;
                return Math.max(0, min - (Math.abs(min) * 0.08 + 0.04));
              },
              (dataMax) => {
                const max = Number(dataMax) || 0;
                return max + (Math.abs(max) * 0.08 + 0.04);
              },
            ]}
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
              `loss, эпоха ${payload?.payload?.epoch}, батч ${payload?.payload?.batch}`,
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
    </div>
  );
}
