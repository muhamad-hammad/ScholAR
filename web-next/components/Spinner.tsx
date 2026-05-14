export default function Spinner({
  className = "",
  label,
}: {
  className?: string;
  label?: string;
}) {
  const ring = (
    <span
      className="inline-block h-4 w-4 shrink-0 animate-spin rounded-full border-2 border-current border-t-transparent"
      role="status"
      aria-label={label ?? "Loading"}
    />
  );

  if (label) {
    return (
      <span className={`inline-flex items-center gap-2 ${className}`}>
        {ring}
        <span className="text-sm text-[var(--app-muted)]">{label}</span>
      </span>
    );
  }

  return (
    <span
      className={`inline-block h-4 w-4 animate-spin rounded-full border-2 border-current border-t-transparent ${className}`}
      role="status"
      aria-label="Loading"
    />
  );
}
