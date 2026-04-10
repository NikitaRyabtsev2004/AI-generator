export function formatNumber(value) {
  return new Intl.NumberFormat('ru-RU').format(Number(value || 0));
}

export function formatDecimal(value, digits = 2) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return '—';
  }

  return Number(value).toFixed(digits);
}

export function formatDateTime(value) {
  if (!value) {
    return '—';
  }

  return new Intl.DateTimeFormat('ru-RU', {
    dateStyle: 'short',
    timeStyle: 'medium',
  }).format(new Date(value));
}

export function formatBoolean(value) {
  return value ? 'Да' : 'Нет';
}

export function previewText(value = '', maxLength = 140) {
  const text = String(value || '').trim();
  if (text.length <= maxLength) {
    return text;
  }

  return `${text.slice(0, maxLength).trimEnd()}...`;
}

export function valueWithUnit(value, unit, digits = 0) {
  if (unit) {
    return `${digits ? formatDecimal(value, digits) : formatNumber(value)} ${unit}`;
  }

  return digits ? formatDecimal(value, digits) : String(value);
}
