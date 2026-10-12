import deviceInfo from '@ohos.deviceInfo';
import { configure, resetCamera, setAutoRotate, setLinxEnabled, setPaused, stats, VolumeStats } from 'libentry.so';

const RENDER_SIZES: number[] = [512, 1024, 1536];
const TARGET_FPS_OPTIONS: number[] = [60, 90, 120];
const SOURCE_URL: string = 'https://livcm.github.io/volumeshader-bm/';

// HarmonyOS exposes no SoC name through any parameter or device-info API; map the publicly
// documented chipsets for known product models, falling back to the product model otherwise.
const SOC_BY_MODEL: Record<string, string> = {
  'WEB-W00': 'Kirin T92',
  'WEB-W09': 'Kirin T92'
};

function socName(): string {
  const mapped = SOC_BY_MODEL[deviceInfo.productModel];
  return mapped !== undefined && mapped !== '' ? mapped : deviceInfo.productModel;
}

function emptyStats(): VolumeStats {
  return {
    realFps: 0,
    gpuMilliseconds: -1,
    realFrames: 0,
    renderSize: 0,
    deviceName: '',
    frameGenerationStatus: '',
    openGtxStatus: '',
    xengineStatus: '',
    qosStatus: '',
    linxStatus: '',
    thermalLevel: 0,
    error: ''
  };
}

@Entry
@Component
struct Index {
  @State stats: VolumeStats = emptyStats();
  @State @Watch('applyConfig') renderSize: number = 1024;
  @State @Watch('applyConfig') optimizedKernel: boolean = true;
  @State @Watch('applyConfig') aiFrameGeneration: boolean = true;
  @State @Watch('applyConfig') openGtx: boolean = true;
  @State @Watch('applyConfig') xengineUpscale: boolean = true;
  @State @Watch('applyLinx') linxEnabled: boolean = true;
  @State @Watch('applyConfig') targetFps: number = 120;
  @State @Watch('applyPause') userPaused: boolean = false;
  @State @Watch('applyPause') pageActive: boolean = false;
  @State autoRotate: boolean = true;
  @State panelOpen: boolean = false;
  @State aboutOpen: boolean = false;
  private pollTimer: number = -1;

  aboutToAppear(): void {
    this.applyConfig();
    this.applyLinx();
    this.pageActive = true;
    this.setPausedNow();
    this.pollTimer = setInterval(() => {
      this.stats = stats();
    }, 500);
  }

  aboutToDisappear(): void {
    if (this.pollTimer >= 0) {
      clearInterval(this.pollTimer);
      this.pollTimer = -1;
    }
    setPaused(true);
  }

  onPageShow(): void {
    this.pageActive = true;
  }

  onPageHide(): void {
    this.pageActive = false;
  }


  private applyLinx(): void {
    setLinxEnabled(this.linxEnabled);
  }
  private applyConfig(): void {
    configure(this.renderSize, this.optimizedKernel, this.aiFrameGeneration, this.openGtx,
      this.xengineUpscale, this.targetFps);
  }

  private applyPause(): void {
    this.setPausedNow();
  }

  private setPausedNow(): void {
    setPaused(this.userPaused || !this.pageActive);
  }
  private gpuText(): string {
    return this.stats.gpuMilliseconds >= 0 ? `${this.stats.gpuMilliseconds.toFixed(2)} ms` : 'n/a';
  }

  private benchmarkBaseline(): void {
    // Fixed original workload: reference kernel, no AI prediction, no OpenGTX.
    this.renderSize = 1024;
    this.optimizedKernel = false;
    this.aiFrameGeneration = false;
    this.openGtx = false;
    this.applyConfig();
  }

  build() {
    Stack({ alignContent: Alignment.TopStart }) {
      XComponent({ id: 'volume', type: XComponentType.SURFACE, libraryname: 'entry' })
        .width('100%')
        .height('100%')

      // Status overlay: every FPS number here is measured on real frames only.
      Column({ space: 2 }) {
        Text(`FPS (measured): ${this.stats.realFps.toFixed(1)}`)
          .fontSize(14)
          .fontColor('#FFFFFF')
        Text(`GPU: ${this.gpuText()}`)
          .fontSize(12)
          .fontColor('#C7C7CC')
        Text(`Render: ${this.stats.renderSize}x${this.stats.renderSize}`)
          .fontSize(12)
          .fontColor('#C7C7CC')
        Text(`Frames: ${Math.floor(this.stats.realFrames)}`)
          .fontSize(12)
          .fontColor('#C7C7CC')
        Text(`Device: ${socName()} · ${this.stats.deviceName === '' ? 'detecting...' : this.stats.deviceName}`)
          .fontSize(12)
          .fontColor('#C7C7CC')
          .maxLines(1)
          .textOverflow({ overflow: TextOverflow.Ellipsis })
        Text(`AI FG: ${this.stats.frameGenerationStatus === '' ? 'unavailable' : this.stats.frameGenerationStatus}` +
          ' (predicted display FPS not measured)')
          .fontSize(12)
          .fontColor('#C7C7CC')
        Text(`OpenGTX: ${this.stats.openGtxStatus === '' ? 'unavailable' : this.stats.openGtxStatus}`)
          .fontSize(12)
          .fontColor('#C7C7CC')
        Text(`XEngine: ${this.stats.xengineStatus === '' ? 'unavailable' : this.stats.xengineStatus}`)
          .fontSize(12)
          .fontColor('#C7C7CC')
        Text(`QoS: ${this.stats.qosStatus === '' ? 'inactive' : this.stats.qosStatus}`)
          .fontSize(12)
          .fontColor('#C7C7CC')
        Text(`Linx: ${this.stats.linxStatus === '' ? 'disabled' : this.stats.linxStatus}`)
          .fontSize(12)
          .fontColor('#C7C7CC')
        Text(`Thermal: ${this.stats.thermalLevel}`)
          .fontSize(12)
          .fontColor('#C7C7CC')
        if (this.stats.error !== '') {
          Text(`Error: ${this.stats.error}`)
            .fontSize(12)
            .fontColor('#FF453A')
            .maxLines(4)
            .textOverflow({ overflow: TextOverflow.Ellipsis })
        }
      }
      .alignItems(HorizontalAlign.Start)
      .padding(8)
      .backgroundColor('#66000000')
      .borderRadius(8)
      .margin(8)
      .hitTestBehavior(HitTestMode.None)

      Button(this.panelOpen ? 'x' : '=')
        .fontSize(16)
        .fontColor('#FFFFFF')
        .backgroundColor('#66000000')
        .width(36)
        .height(36)
        .borderRadius(18)
        .position({ x: '100%', y: 8 })
        .markAnchor({ x: '100%', y: 0 })
        .margin({ right: 8 })
        .onClick(() => {
          this.panelOpen = !this.panelOpen;
        })

      if (this.panelOpen) {
        Scroll() {
          Column({ space: 8 }) {
            Text('CONFIG')
              .fontSize(12)
              .fontColor('#8E8E93')
            Row({ space: 8 }) {
              Text('Kernel')
                .fontSize(14)
                .fontColor('#FFFFFF')
                .layoutWeight(1)
              Toggle({ type: ToggleType.Switch, isOn: this.optimizedKernel })
                .onChange((isOn: boolean) => {
                  this.optimizedKernel = isOn;
                })
            }
            .width('100%')
            Text(this.optimizedKernel ? 'Arithmetic optimized kernel' : 'Original reference kernel (baseline)')
              .fontSize(11)
              .fontColor('#8E8E93')

            Row({ space: 8 }) {
              Text('Render size')
                .fontSize(14)
                .fontColor('#FFFFFF')
                .layoutWeight(1)
              ForEach(RENDER_SIZES, (size: number) => {
                Button(`${size}`)
                  .fontSize(12)
                  .height(28)
                  .padding({ left: 10, right: 10 })
                  .fontColor(this.renderSize === size ? '#FFFFFF' : '#C7C7CC')
                  .backgroundColor(this.renderSize === size ? '#0A84FF' : '#2C2C2E')
                  .borderRadius(14)
                  .onClick(() => {
                    this.renderSize = size;
                  })
              }, (size: number) => size.toString())
            }
            .width('100%')

            Row({ space: 8 }) {
              Text('Target FPS')
                .fontSize(14)
                .fontColor('#FFFFFF')
                .layoutWeight(1)
              ForEach(TARGET_FPS_OPTIONS, (fps: number) => {
                Button(`${fps}`)
                  .fontSize(12)
                  .height(28)
                  .padding({ left: 10, right: 10 })
                  .fontColor(this.targetFps === fps ? '#FFFFFF' : '#C7C7CC')
                  .backgroundColor(this.targetFps === fps ? '#0A84FF' : '#2C2C2E')
                  .borderRadius(14)
                  .onClick(() => {
                    this.targetFps = fps;
                  })
              }, (fps: number) => fps.toString())
            }
            .width('100%')

            Row({ space: 8 }) {
              Text('AI frame generation')
                .fontSize(14)
                .fontColor('#FFFFFF')
                .layoutWeight(1)
              Toggle({ type: ToggleType.Switch, isOn: this.aiFrameGeneration })
                .onChange((isOn: boolean) => {
                  this.aiFrameGeneration = isOn;
                })
            }
            .width('100%')

            Row({ space: 8 }) {
              Text('OpenGTX')
                .fontSize(14)
                .fontColor('#FFFFFF')
                .layoutWeight(1)
              Toggle({ type: ToggleType.Switch, isOn: this.openGtx })
                .onChange((isOn: boolean) => {
                  this.openGtx = isOn;
                })
            }
            .width('100%')

            Row({ space: 8 }) {
              Text('XEngine upscale')
                .fontSize(14)
                .fontColor('#FFFFFF')
                .layoutWeight(1)
              Toggle({ type: ToggleType.Switch, isOn: this.xengineUpscale })
                .onChange((isOn: boolean) => {
                  this.xengineUpscale = isOn;
                })
            }
            .width('100%')
            Text('Maleoon GPU spatial upscaler replaces linear sampling below native size')
              .fontSize(11)
              .fontColor('#8E8E93')

            Row({ space: 8 }) {
              Text('Linx hotspot')
                .fontSize(14)
                .fontColor('#FFFFFF')
                .layoutWeight(1)
              Toggle({ type: ToggleType.Switch, isOn: this.linxEnabled })
                .onChange((isOn: boolean) => {
                  this.linxEnabled = isOn;
                })
            }
            .width('100%')
            Text('API 26+, requires AppGallery release build')
              .fontSize(11)
              .fontColor('#8E8E93')

            Row({ space: 8 }) {
              Text('Paused')
                .fontSize(14)
                .fontColor('#FFFFFF')
                .layoutWeight(1)
              Toggle({ type: ToggleType.Switch, isOn: this.userPaused })
                .onChange((isOn: boolean) => {
                  this.userPaused = isOn;
                })
            }
            .width('100%')

            Row({ space: 8 }) {
              Text('Auto-rotate')
                .fontSize(14)
                .fontColor('#FFFFFF')
                .layoutWeight(1)
              Toggle({ type: ToggleType.Switch, isOn: this.autoRotate })
                .onChange((isOn: boolean) => {
                  this.autoRotate = isOn;
                  setAutoRotate(isOn);
                })
            }
            .width('100%')

            Row({ space: 8 }) {
              Button('Reset camera')
                .fontSize(12)
                .height(30)
                .fontColor('#FFFFFF')
                .backgroundColor('#2C2C2E')
                .borderRadius(15)
                .onClick(() => {
                  resetCamera();
                })
              Button('Benchmark baseline')
                .fontSize(12)
                .height(30)
                .fontColor('#FFD60A')
                .backgroundColor('#2C2C2E')
                .borderRadius(15)
                .onClick(() => {
                  this.benchmarkBaseline();
                })
            }
            .width('100%')

            Text('About')
              .fontSize(12)
              .fontColor('#0A84FF')
              .onClick(() => {
                this.aboutOpen = !this.aboutOpen;
              })
            if (this.aboutOpen) {
              Text(`Vulkan port of the browser Mandelbulb shader.\nSource: ${SOURCE_URL}\n` +
                'Benchmark baseline = 1024, original kernel, AI and OpenGTX off.\n' +
                'Only measured real-frame FPS is reported; AI predicted display frames are not counted.')
                .fontSize(11)
                .fontColor('#8E8E93')
            }
          }
          .alignItems(HorizontalAlign.Start)
          .padding(12)
          .width('100%')
        }
        .constraintSize({ maxHeight: '45%' })
        .width('94%')
        .backgroundColor('#CC1C1C1E')
        .borderRadius(16)
        .position({ x: '3%', y: '100%' })
        .markAnchor({ x: 0, y: '100%' })
        .margin({ bottom: 8 })
      }
    }
    .width('100%')
    .height('100%')
    .backgroundColor('#000000')
  }
}
