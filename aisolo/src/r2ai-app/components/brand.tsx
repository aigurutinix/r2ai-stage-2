import { cn } from "@/lib/utils";

/** Logo cá voi FinWhale + wordmark gradient cyan→xanh. */
export function Brand({
  iconSize = 32,
  textClass,
  className,
}: {
  iconSize?: number;
  textClass?: string;
  className?: string;
}) {
  return (
    <div className={cn("flex items-center gap-2", className)}>
      {/* eslint-disable-next-line @next/next/no-img-element */}
      <img
        src="/logo.png"
        alt="FinWhale"
        style={{ width: iconSize, height: iconSize }}
        className="shrink-0 object-contain"
      />
      <span
        className={cn(
          "bg-gradient-to-r from-[#10b4d6] to-[#1c64e8] bg-clip-text font-bold tracking-tight text-transparent",
          textClass,
        )}
      >
        FinWhale
      </span>
    </div>
  );
}
