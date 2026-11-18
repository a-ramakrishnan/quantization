import { useEffect, useState } from "react";

const CHAPTER_SECONDS = 6;
const SCRUB_STEP = 0.05;

export function useLessonPlayback(chapterCount: number, enabled = true, speed = 1, chapterSeconds = CHAPTER_SECONDS) {
  const duration = chapterCount * chapterSeconds;
  const [time, setTime] = useState(0);
  const [end, setEnd] = useState(duration);
  const [playing, setPlaying] = useState(false);
  const [reducedMotion, setReducedMotion] = useState(false);

  useEffect(() => {
    const media = window.matchMedia("(prefers-reduced-motion: reduce)");
    const update = () => {
      setReducedMotion(media.matches);
      if (media.matches) setPlaying(false);
    };
    update();
    media.addEventListener("change", update);
    return () => media.removeEventListener("change", update);
  }, []);

  useEffect(() => {
    if (!playing || !enabled || reducedMotion) return;
    let frame: number;
    let previous = performance.now();
    const tick = (now: number) => {
      const elapsed = Math.max(0, Math.min(now - previous, 100)) / 1000;
      previous = now;
      setTime(current => Math.min(end, current + elapsed * speed));
      frame = requestAnimationFrame(tick);
    };
    frame = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(frame);
  }, [playing, enabled, reducedMotion, speed, end]);

  useEffect(() => {
    if (time >= end) setPlaying(false);
  }, [time, end]);

  const chapter = Math.min(chapterCount - 1, Math.floor(time / chapterSeconds));

  function seek(nextTime: number) {
    setTime(nextTime);
    setPlaying(false);
  }

  function selectChapter(index: number) {
    const start = index * chapterSeconds;
    const lastFrame = start + chapterSeconds - SCRUB_STEP;
    setEnd(lastFrame);
    setTime(reducedMotion ? lastFrame : start);
    setPlaying(!reducedMotion);
  }

  function togglePlayback() {
    if (playing) {
      setPlaying(false);
      return;
    }
    if (time >= duration) setTime(0);
    setEnd(duration);
    setPlaying(true);
  }

  return {
    chapter, time, duration, playing, reducedMotion, scrubStep: SCRUB_STEP,
    chapterTime: time - chapter * chapterSeconds,
    seek, selectChapter, togglePlayback,
  };
}
