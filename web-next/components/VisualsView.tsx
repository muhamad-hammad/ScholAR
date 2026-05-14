"use client";

import { useEffect, useState, useCallback } from "react";
import { useAppState } from "../lib/useAppState";
import { extractImages } from "../lib/api";
import Spinner from "./Spinner";

export default function VisualsView() {
  const { pdfFile, pdfImages, setPdfImages } = useAppState();
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [lightboxIndex, setLightboxIndex] = useState<number | null>(null);

  const closeLightbox = useCallback(() => setLightboxIndex(null), []);

  const visibleImages = pdfImages?.filter((img) => img.page !== 1) ?? null;

  const goNext = useCallback(() => {
    if (visibleImages && lightboxIndex !== null)
      setLightboxIndex((lightboxIndex + 1) % visibleImages.length);
  }, [visibleImages, lightboxIndex]);

  const goPrev = useCallback(() => {
    if (visibleImages && lightboxIndex !== null)
      setLightboxIndex((lightboxIndex - 1 + visibleImages.length) % visibleImages.length);
  }, [visibleImages, lightboxIndex]);

  useEffect(() => {
    if (lightboxIndex === null) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") closeLightbox();
      if (e.key === "ArrowRight") goNext();
      if (e.key === "ArrowLeft") goPrev();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [lightboxIndex, closeLightbox, goNext, goPrev]);

  useEffect(() => {
    if (!pdfFile || pdfImages !== null) return;

    setError(null);
    setLoading(true);
    extractImages(pdfFile)
      .then((result) => setPdfImages(result.images))
      .catch((err) => {
        setError(err instanceof Error ? err.message : "An error occurred.");
        setPdfImages([]);
      })
      .finally(() => setLoading(false));
  }, [pdfFile, pdfImages, setPdfImages]);

  const heading = (
    <div>
      <h2 className="text-xl font-semibold">Paper visualizations</h2>
      <p className="mt-1 text-sm text-[var(--app-muted)]">
        Figures, charts, and other images extracted from the indexed PDF.
      </p>
    </div>
  );

  if (!pdfFile) {
    return (
      <div className="flex flex-col gap-4">
        {heading}
        <div className="rounded-md border border-blue-500/30 bg-blue-500/10 px-4 py-3 text-sm text-blue-400">
          Upload a PDF and run ingestion to see its figures here.
        </div>
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-4">
      {heading}

      {loading && (
        <div className="py-4">
          <Spinner label="Extracting images from the PDF…" />
        </div>
      )}

      {error && (
        <div className="rounded-md border border-red-500/30 bg-red-500/10 px-4 py-3 text-sm text-red-400">
          {error}
        </div>
      )}

      {visibleImages !== null && !loading && (
        <>
          {visibleImages.length === 0 ? (
            <div className="rounded-md border border-blue-500/30 bg-blue-500/10 px-4 py-3 text-sm text-blue-400">
              No embedded images were found in this PDF.
            </div>
          ) : (
            <>
              <div className="rounded-md border border-green-500/30 bg-green-500/10 px-4 py-3 text-sm text-green-400">
                Found {visibleImages.length} image(s) in the PDF.
              </div>
              <div className="flex flex-row gap-4 overflow-x-auto pb-2">
                {visibleImages.map((img, i) => (
                  <button
                    key={i}
                    type="button"
                    onClick={() => setLightboxIndex(i)}
                    className="flex shrink-0 w-72 flex-col gap-1 text-left focus:outline-none focus-visible:ring-2 focus-visible:ring-[var(--app-accent)]"
                  >
                    <img
                      src={`data:image/${img.ext};base64,${img.data}`}
                      alt={`Page ${img.page}`}
                      className="w-72 h-64 rounded border border-[var(--app-border)] object-contain transition-opacity hover:opacity-80 cursor-zoom-in"
                    />
                    <p className="text-center text-xs text-[var(--app-muted)]">
                      Page {img.page}
                    </p>
                  </button>
                ))}
              </div>

              {lightboxIndex !== null && (
                <div
                  className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-sm"
                  onClick={closeLightbox}
                >
                  <button
                    type="button"
                    onClick={(e) => { e.stopPropagation(); goPrev(); }}
                    className="absolute left-4 top-1/2 -translate-y-1/2 rounded-full bg-black/60 px-3 py-2 text-white text-lg hover:bg-black/80"
                    aria-label="Previous image"
                  >
                    ‹
                  </button>

                  <div
                    className="relative flex max-h-[90vh] max-w-[90vw] flex-col items-center gap-2"
                    onClick={(e) => e.stopPropagation()}
                  >
                    <img
                      src={`data:image/${visibleImages[lightboxIndex].ext};base64,${visibleImages[lightboxIndex].data}`}
                      alt={`Page ${visibleImages[lightboxIndex].page}`}
                      className="max-h-[85vh] max-w-[85vw] rounded border border-[var(--app-border)] object-contain"
                    />
                    <p className="text-sm text-white/70">
                      {lightboxIndex + 1} / {visibleImages.length} — Page {visibleImages[lightboxIndex].page}
                    </p>
                  </div>

                  <button
                    type="button"
                    onClick={(e) => { e.stopPropagation(); goNext(); }}
                    className="absolute right-4 top-1/2 -translate-y-1/2 rounded-full bg-black/60 px-3 py-2 text-white text-lg hover:bg-black/80"
                    aria-label="Next image"
                  >
                    ›
                  </button>

                  <button
                    type="button"
                    onClick={closeLightbox}
                    className="absolute right-4 top-4 rounded-full bg-black/60 px-2 py-1 text-white text-sm hover:bg-black/80"
                    aria-label="Close"
                  >
                    ✕
                  </button>
                </div>
              )}
            </>
          )}

          <button
            type="button"
            onClick={() => setPdfImages(null)}
            className="w-fit rounded border border-[var(--app-border)] bg-[var(--app-btn-bg)] px-4 py-2 text-sm font-medium text-[var(--app-text)] transition-colors hover:bg-[var(--app-btn-hover-bg)]"
          >
            Re-extract images
          </button>
        </>
      )}
    </div>
  );
}
