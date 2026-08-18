# Dataset review previews are bounded and schema-neutral

LeRobot's canonical video encoding and its review surface have different jobs. The dataset needs
storage-efficient, training-ready bytes; a W&B Workspace needs browser-playable media with stable
keys. LeRobot deliberately defaults to AV1 after benchmarking it against H.264 and H.265
([benchmark](https://github.com/huggingface/lerobot/pull/282),
[adoption](https://github.com/huggingface/lerobot/pull/302)), while later work retained that default
but exposed other codecs for recording-time CPU and hardware trade-offs
([codec selection](https://github.com/huggingface/lerobot/pull/2771),
[streaming and hardware encoders](https://github.com/huggingface/lerobot/pull/2974)). The W&B
integration must not make users choose their canonical codec for the convenience of its preview UI.

Dataset Artifacts therefore keep their original bytes. Review media is planned through one
schema-neutral episode interface and, when needed, emitted as a separate compact derivative:

- v2.1 resolves an episode to its episode-per-file source.
- v3 resolves the same logical request to a shared source file plus exact start/end timestamps.
- The default selection is one deterministic, exact episode and its cameras in both schemas. It is
  logged under `dataset_video/representative/<camera>` so a Workspace configured from v3 runs also
  works for a v2.1 upload. The actual episode index and dataset schema version remain explicit run
  metadata.
- Explicit and all-episode selections use `dataset_video/episode_<index>/<camera>` for both schemas.
  Camera keys remain reversible and are never guessed or aliased across renamed cameras.

The derivative profile is H.264/yuv420p MP4, at most 640 pixels wide and 15 frames per second, CRF
32, with a two-second GOP. CRF and GOP are deterministic settings for derivatives we encode; they
are not source-eligibility predicates because an existing container cannot reliably recover those
encoder settings. It retains each selected episode's full duration and every selected camera. An
exact-episode H.264/yuv420p MP4 may take the fast path when its observable container, geometry, and
frame rate fit the browser constraints and the complete preview batch fits its byte budget. This
normally permits v2.1 H.264 episode files to take the fast path. A v3 shared file still needs exact
slicing unless metadata proves that the file contains only the requested episode. The hard aggregate
byte budget bounds the upload even when source CRF or GOP cannot be observed.

The preview batch has two independent guards: `--preview-all` keeps its default 50-episode ceiling,
and the prepared aggregate of Run Media bytes must not exceed the smaller of 250 MiB and 20% of the
canonical dataset directory size. All sources are resolved, previews prepared, and bytes measured
before `wandb.init`.
Exceeding a guard fails with the measured total and remediation choices; the tool does not silently
shorten episodes, discard cameras, or progressively reduce quality.

The Artifact-backed playback spike is complete. Against the W&B SDK 0.27.2, there is no supported
zero-copy path for a standalone Artifact video to appear as Workspace Run Media: Artifact files and
Run Media remain separate publication paths. The reference-video work supports remote references
inside Artifacts but explicitly does not bind those references to a run
([wandb/wandb#11412](https://github.com/wandb/wandb/pull/11412)), and Artifact Tables are a distinct
UI contract rather than a replacement for Run Media. The integration therefore uses bounded Run
Media fallback. Eligible v2.1 H.264 previews can still reference the canonical bytes; the same
profile and budget checks apply. v3 episode slicing remains necessary.

Alternatives rejected:

- Changing the canonical dataset to H.264 for W&B. It couples training storage to one review UI and
  still does not solve v3 episode boundaries or the Artifact/Run Media separation.
- GIF previews. GIF is palette-limited and compresses frames with LZW rather than modern inter-frame
  prediction; controlled samples were substantially larger than compact H.264 while looking worse.
- Relying only on the episode-count ceiling. Camera count, resolution, frame rate, and episode
  duration can make two uploads with the same episode count differ by orders of magnitude.
- Automatic degradation after crossing the budget. It makes review quality data-dependent and
  hides what was discarded; an explicit failure is reproducible and auditable.
