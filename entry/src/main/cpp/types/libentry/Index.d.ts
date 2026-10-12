export interface VolumeStats {
  realFps: number;
  gpuMilliseconds: number;
  realFrames: number;
  renderSize: number;
  deviceName: string;
  frameGenerationStatus: string;
  openGtxStatus: string;
  xengineStatus: string;
  qosStatus: string;
  linxStatus: string;
  thermalLevel: number;
  error: string;
}

export const configure: (renderSize: number, optimizedKernel: boolean, aiFrameGeneration: boolean,
  openGtx: boolean, xengineUpscale: boolean, targetFps: number) => void;

export const setPaused: (paused: boolean) => void;

export const setForeground: (foreground: boolean) => void;

export const setLinxEnabled: (enabled: boolean) => void;

export const setAutoRotate: (enabled: boolean) => void;

export const resetCamera: () => void;

export const stats: () => VolumeStats;
