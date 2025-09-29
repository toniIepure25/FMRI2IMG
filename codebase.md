# .dvc/.gitignore

```
/config.local
/tmp
/cache

```

# .dvc/cache/files/md5/2c/fdc695fc4d02a0aef4e9c1185bf1ae

```
image_id,caption
0,a dog on grass
1,a red car
2,a mountain scene

```

# .dvc/cache/files/md5/3e/02189797c8eb27e07307bbd93b4667

```
paths:
  data_dir: ${oc.env:DATA_DIR, ./data}
  raw_dir: ${paths.data_dir}/raw
  proc_dir: ${paths.data_dir}/processed
  artifacts_dir: ${paths.data_dir}/artifacts
data:
  name: nsd
  subjects:
  - subj01
  split:
    train_ratio: 0.9
    seed: 42
  paths:
    images_root: ${paths.raw_dir}/nsd/images
    fmri_root: ${paths.raw_dir}/nsd/fmri
    captions: ${paths.raw_dir}/nsd/captions.csv
  fmriprep:
    bids_root: ${paths.raw_dir}/nsd/bids
    out_root: ${paths.proc_dir}/nsd/fmriprep
    fs_license: ${oc.env:FS_LICENSE, ./licenses/freesurfer_license.txt}
  roi:
    mode: atlas
    atlas: glasser
    vector_dim: 2048
    out_dir: ${paths.proc_dir}/nsd/roi
train:
  max_epochs: 1
  batch_size: 2
  num_workers: 4
  precision: 32
  optimizer:
    lr: 0.001
  model:
    fmri_input_dim: 2048
    latent_dim: 768
    hidden:
    - 2048
    - 1024
    - 768
eval:
  metrics:
  - ssim
  - psnr
  - clip_score
wandb:
  enabled: true
  project: ${oc.env:WANDB_PROJECT, fmri2image}
  entity: ${oc.env:WANDB_ENTITY, null}
  mode: online
seed: 1337
device: cuda
run:
  name: baseline_train
  output_dir: outputs/${now:%Y-%m-%d}/${run.name}

```

# .dvc/cache/files/md5/4c/dec22ee97a0a9081773527817c0b85

```
hydra:
  run:
    dir: outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}
  sweep:
    dir: multirun/${now:%Y-%m-%d}/${now:%H-%M-%S}
    subdir: ${hydra.job.num}
  launcher:
    _target_: hydra._internal.core_plugins.basic_launcher.BasicLauncher
  sweeper:
    _target_: hydra._internal.core_plugins.basic_sweeper.BasicSweeper
    max_batch_size: null
    params: null
  help:
    app_name: ${hydra.job.name}
    header: '${hydra.help.app_name} is powered by Hydra.

      '
    footer: 'Powered by Hydra (https://hydra.cc)

      Use --hydra-help to view Hydra specific help

      '
    template: '${hydra.help.header}

      == Configuration groups ==

      Compose your configuration from those groups (group=option)


      $APP_CONFIG_GROUPS


      == Config ==

      Override anything in the config (foo.bar=value)


      $CONFIG


      ${hydra.help.footer}

      '
  hydra_help:
    template: 'Hydra (${hydra.runtime.version})

      See https://hydra.cc for more info.


      == Flags ==

      $FLAGS_HELP


      == Configuration groups ==

      Compose your configuration from those groups (For example, append hydra/job_logging=disabled
      to command line)


      $HYDRA_CONFIG_GROUPS


      Use ''--cfg hydra'' to Show the Hydra config.

      '
    hydra_help: ???
  hydra_logging:
    version: 1
    formatters:
      simple:
        format: '[%(asctime)s][HYDRA] %(message)s'
    handlers:
      console:
        class: logging.StreamHandler
        formatter: simple
        stream: ext://sys.stdout
    root:
      level: INFO
      handlers:
      - console
    loggers:
      logging_example:
        level: DEBUG
    disable_existing_loggers: false
  job_logging:
    version: 1
    formatters:
      simple:
        format: '[%(asctime)s][%(name)s][%(levelname)s] - %(message)s'
    handlers:
      console:
        class: logging.StreamHandler
        formatter: simple
        stream: ext://sys.stdout
      file:
        class: logging.FileHandler
        formatter: simple
        filename: ${hydra.runtime.output_dir}/${hydra.job.name}.log
    root:
      level: INFO
      handlers:
      - console
      - file
    disable_existing_loggers: false
  env: {}
  mode: RUN
  searchpath: []
  callbacks: {}
  output_subdir: .hydra
  overrides:
    hydra:
    - hydra.mode=RUN
    task:
    - ++run.name=baseline_train
    - train=baseline
  job:
    name: cli
    chdir: null
    override_dirname: ++run.name=baseline_train,train=baseline
    id: ???
    num: ???
    config_name: config
    env_set: {}
    env_copy: []
    config:
      override_dirname:
        kv_sep: '='
        item_sep: ','
        exclude_keys: []
  runtime:
    version: 1.3.2
    version_base: '1.3'
    cwd: /home/tonystark/Desktop/Bachelor/fmri2image
    config_sources:
    - path: hydra.conf
      schema: pkg
      provider: hydra
    - path: /home/tonystark/Desktop/Bachelor/fmri2image/configs
      schema: file
      provider: main
    - path: ''
      schema: structured
      provider: schema
    output_dir: /home/tonystark/Desktop/Bachelor/fmri2image/outputs/2025-09-28/23-32-44
    choices:
      wandb: wandb
      eval: default
      train: baseline
      data: nsd
      paths: paths
      hydra/env: default
      hydra/callbacks: null
      hydra/job_logging: default
      hydra/hydra_logging: default
      hydra/hydra_help: default
      hydra/help: default
      hydra/sweeper: basic
      hydra/launcher: basic
      hydra/output: default
  verbose: false

```

# .dvc/cache/files/md5/11/290f9045b213fc093bcd600306a7e0

```
[2025-09-28 23:45:30,844][__main__][INFO] - Config:
paths:
  data_dir: ${oc.env:DATA_DIR, ./data}
  raw_dir: ${paths.data_dir}/raw
  proc_dir: ${paths.data_dir}/processed
  artifacts_dir: ${paths.data_dir}/artifacts
data:
  name: nsd
  subjects:
  - subj01
  split:
    train_ratio: 0.9
    seed: 42
  paths:
    images_root: ${paths.raw_dir}/nsd/images
    fmri_root: ${paths.raw_dir}/nsd/fmri
    captions: ${paths.raw_dir}/nsd/captions.csv
  fmriprep:
    bids_root: ${paths.raw_dir}/nsd/bids
    out_root: ${paths.proc_dir}/nsd/fmriprep
    fs_license: ${oc.env:FS_LICENSE, ./licenses/freesurfer_license.txt}
  roi:
    mode: atlas
    atlas: glasser
    vector_dim: 2048
    out_dir: ${paths.proc_dir}/nsd/roi
train:
  max_epochs: 1
  batch_size: 2
  num_workers: 4
  precision: 32
  optimizer:
    lr: 0.001
  model:
    fmri_input_dim: 2048
    latent_dim: 768
    hidden:
    - 2048
    - 1024
    - 768
eval:
  metrics:
  - ssim
  - psnr
  - clip_score
wandb:
  enabled: true
  project: ${oc.env:WANDB_PROJECT, fmri2image}
  entity: ${oc.env:WANDB_ENTITY, null}
  mode: online
seed: 1337
device: cuda
run:
  name: baseline_train
  output_dir: outputs/${now:%Y-%m-%d}/${run.name}


```

# .dvc/cache/files/md5/20/0e4aa621588a1d459d72e00ab23fe1.dir

```dir
[{"md5": "4274adec5d9eaefdddf3e352e03fba64", "relpath": "nsd/README.txt"}, {"md5": "2cfdc695fc4d02a0aef4e9c1185bf1ae", "relpath": "nsd/captions.csv"}]
```

# .dvc/cache/files/md5/25/77ce444d57eb6c9b162b4a4c8b09f4

This is a binary file of the type: Binary

# .dvc/cache/files/md5/37/f1c8f38e0280f7823b6dcf612120e5.dir

```dir
[{"md5": "71407b493ccfaf99e32267320717b0e9", "relpath": "2025-09-28/23-32-44/.hydra/config.yaml"}, {"md5": "4cdec22ee97a0a9081773527817c0b85", "relpath": "2025-09-28/23-32-44/.hydra/hydra.yaml"}, {"md5": "73bc718bee9b637ae12f8173a862007a", "relpath": "2025-09-28/23-32-44/.hydra/overrides.yaml"}, {"md5": "f83fdd2b68250633178df51bd1188267", "relpath": "2025-09-28/23-32-44/cli.log"}]
```

# .dvc/cache/files/md5/42/74adec5d9eaefdddf3e352e03fba64

```
NSD placeholder. Pentru date reale: configurați AWS CLI și sincronizați din bucketul NSD.
Struc.: raw/nsd/images/, raw/nsd/fmri/, raw/nsd/captions.csv

```

# .dvc/cache/files/md5/47/c9fb596d852b19d0a97abc4504a6ae

This is a binary file of the type: Binary

# .dvc/cache/files/md5/60/81bbb2697282d0d75d61047f8b0246

```
hydra:
  run:
    dir: outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}
  sweep:
    dir: multirun/${now:%Y-%m-%d}/${now:%H-%M-%S}
    subdir: ${hydra.job.num}
  launcher:
    _target_: hydra._internal.core_plugins.basic_launcher.BasicLauncher
  sweeper:
    _target_: hydra._internal.core_plugins.basic_sweeper.BasicSweeper
    max_batch_size: null
    params: null
  help:
    app_name: ${hydra.job.name}
    header: '${hydra.help.app_name} is powered by Hydra.

      '
    footer: 'Powered by Hydra (https://hydra.cc)

      Use --hydra-help to view Hydra specific help

      '
    template: '${hydra.help.header}

      == Configuration groups ==

      Compose your configuration from those groups (group=option)


      $APP_CONFIG_GROUPS


      == Config ==

      Override anything in the config (foo.bar=value)


      $CONFIG


      ${hydra.help.footer}

      '
  hydra_help:
    template: 'Hydra (${hydra.runtime.version})

      See https://hydra.cc for more info.


      == Flags ==

      $FLAGS_HELP


      == Configuration groups ==

      Compose your configuration from those groups (For example, append hydra/job_logging=disabled
      to command line)


      $HYDRA_CONFIG_GROUPS


      Use ''--cfg hydra'' to Show the Hydra config.

      '
    hydra_help: ???
  hydra_logging:
    version: 1
    formatters:
      simple:
        format: '[%(asctime)s][HYDRA] %(message)s'
    handlers:
      console:
        class: logging.StreamHandler
        formatter: simple
        stream: ext://sys.stdout
    root:
      level: INFO
      handlers:
      - console
    loggers:
      logging_example:
        level: DEBUG
    disable_existing_loggers: false
  job_logging:
    version: 1
    formatters:
      simple:
        format: '[%(asctime)s][%(name)s][%(levelname)s] - %(message)s'
    handlers:
      console:
        class: logging.StreamHandler
        formatter: simple
        stream: ext://sys.stdout
      file:
        class: logging.FileHandler
        formatter: simple
        filename: ${hydra.runtime.output_dir}/${hydra.job.name}.log
    root:
      level: INFO
      handlers:
      - console
      - file
    disable_existing_loggers: false
  env: {}
  mode: RUN
  searchpath: []
  callbacks: {}
  output_subdir: .hydra
  overrides:
    hydra:
    - hydra.mode=RUN
    task:
    - ++run.name=baseline_train
    - train=baseline
  job:
    name: cli
    chdir: null
    override_dirname: ++run.name=baseline_train,train=baseline
    id: ???
    num: ???
    config_name: config
    env_set: {}
    env_copy: []
    config:
      override_dirname:
        kv_sep: '='
        item_sep: ','
        exclude_keys: []
  runtime:
    version: 1.3.2
    version_base: '1.3'
    cwd: /home/tonystark/Desktop/Bachelor/fmri2image
    config_sources:
    - path: hydra.conf
      schema: pkg
      provider: hydra
    - path: /home/tonystark/Desktop/Bachelor/fmri2image/configs
      schema: file
      provider: main
    - path: ''
      schema: structured
      provider: schema
    output_dir: /home/tonystark/Desktop/Bachelor/fmri2image/outputs/2025-09-29/00-23-45
    choices:
      wandb: wandb
      eval: default
      train: baseline
      data: nsd
      paths: paths
      hydra/env: default
      hydra/callbacks: null
      hydra/job_logging: default
      hydra/hydra_logging: default
      hydra/hydra_help: default
      hydra/help: default
      hydra/sweeper: basic
      hydra/launcher: basic
      hydra/output: default
  verbose: false

```

# .dvc/cache/files/md5/66/67798a2b9efe620802db99d520b5bf.dir

```dir
[{"md5": "3e02189797c8eb27e07307bbd93b4667", "relpath": "2025-09-28/23-45-30/.hydra/config.yaml"}, {"md5": "81950fa78e5a3676cb85243e02f75211", "relpath": "2025-09-28/23-45-30/.hydra/hydra.yaml"}, {"md5": "73bc718bee9b637ae12f8173a862007a", "relpath": "2025-09-28/23-45-30/.hydra/overrides.yaml"}, {"md5": "11290f9045b213fc093bcd600306a7e0", "relpath": "2025-09-28/23-45-30/cli.log"}]
```

# .dvc/cache/files/md5/71/407b493ccfaf99e32267320717b0e9

```
paths:
  data_dir: ${oc.env:DATA_DIR, ./data}
  raw_dir: ${paths.data_dir}/raw
  proc_dir: ${paths.data_dir}/processed
  artifacts_dir: ${paths.data_dir}/artifacts
data:
  name: nsd
  subjects:
  - subj01
  split:
    train_ratio: 0.9
    seed: 42
  paths:
    images_root: ${paths.raw_dir}/nsd/images
    fmri_root: ${paths.raw_dir}/nsd/fmri
    captions: ${paths.raw_dir}/nsd/captions.csv
train:
  max_epochs: 1
  batch_size: 2
  num_workers: 4
  precision: 32
  optimizer:
    lr: 0.001
  model:
    fmri_input_dim: 2048
    latent_dim: 768
    hidden:
    - 2048
    - 1024
    - 768
eval:
  metrics:
  - ssim
  - psnr
  - clip_score
wandb:
  enabled: true
  project: ${oc.env:WANDB_PROJECT, fmri2image}
  entity: ${oc.env:WANDB_ENTITY, null}
  mode: online
seed: 1337
device: cuda
run:
  name: baseline_train
  output_dir: outputs/${now:%Y-%m-%d}/${run.name}

```

# .dvc/cache/files/md5/73/bc718bee9b637ae12f8173a862007a

```
- ++run.name=baseline_train
- train=baseline

```

# .dvc/cache/files/md5/79/a62e2ffa10ebce2562583f42158d68.dir

```dir
[{"md5": "4274adec5d9eaefdddf3e352e03fba64", "relpath": "README.txt"}, {"md5": "2cfdc695fc4d02a0aef4e9c1185bf1ae", "relpath": "captions.csv"}]
```

# .dvc/cache/files/md5/81/950fa78e5a3676cb85243e02f75211

```
hydra:
  run:
    dir: outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}
  sweep:
    dir: multirun/${now:%Y-%m-%d}/${now:%H-%M-%S}
    subdir: ${hydra.job.num}
  launcher:
    _target_: hydra._internal.core_plugins.basic_launcher.BasicLauncher
  sweeper:
    _target_: hydra._internal.core_plugins.basic_sweeper.BasicSweeper
    max_batch_size: null
    params: null
  help:
    app_name: ${hydra.job.name}
    header: '${hydra.help.app_name} is powered by Hydra.

      '
    footer: 'Powered by Hydra (https://hydra.cc)

      Use --hydra-help to view Hydra specific help

      '
    template: '${hydra.help.header}

      == Configuration groups ==

      Compose your configuration from those groups (group=option)


      $APP_CONFIG_GROUPS


      == Config ==

      Override anything in the config (foo.bar=value)


      $CONFIG


      ${hydra.help.footer}

      '
  hydra_help:
    template: 'Hydra (${hydra.runtime.version})

      See https://hydra.cc for more info.


      == Flags ==

      $FLAGS_HELP


      == Configuration groups ==

      Compose your configuration from those groups (For example, append hydra/job_logging=disabled
      to command line)


      $HYDRA_CONFIG_GROUPS


      Use ''--cfg hydra'' to Show the Hydra config.

      '
    hydra_help: ???
  hydra_logging:
    version: 1
    formatters:
      simple:
        format: '[%(asctime)s][HYDRA] %(message)s'
    handlers:
      console:
        class: logging.StreamHandler
        formatter: simple
        stream: ext://sys.stdout
    root:
      level: INFO
      handlers:
      - console
    loggers:
      logging_example:
        level: DEBUG
    disable_existing_loggers: false
  job_logging:
    version: 1
    formatters:
      simple:
        format: '[%(asctime)s][%(name)s][%(levelname)s] - %(message)s'
    handlers:
      console:
        class: logging.StreamHandler
        formatter: simple
        stream: ext://sys.stdout
      file:
        class: logging.FileHandler
        formatter: simple
        filename: ${hydra.runtime.output_dir}/${hydra.job.name}.log
    root:
      level: INFO
      handlers:
      - console
      - file
    disable_existing_loggers: false
  env: {}
  mode: RUN
  searchpath: []
  callbacks: {}
  output_subdir: .hydra
  overrides:
    hydra:
    - hydra.mode=RUN
    task:
    - ++run.name=baseline_train
    - train=baseline
  job:
    name: cli
    chdir: null
    override_dirname: ++run.name=baseline_train,train=baseline
    id: ???
    num: ???
    config_name: config
    env_set: {}
    env_copy: []
    config:
      override_dirname:
        kv_sep: '='
        item_sep: ','
        exclude_keys: []
  runtime:
    version: 1.3.2
    version_base: '1.3'
    cwd: /home/tonystark/Desktop/Bachelor/fmri2image
    config_sources:
    - path: hydra.conf
      schema: pkg
      provider: hydra
    - path: /home/tonystark/Desktop/Bachelor/fmri2image/configs
      schema: file
      provider: main
    - path: ''
      schema: structured
      provider: schema
    output_dir: /home/tonystark/Desktop/Bachelor/fmri2image/outputs/2025-09-28/23-45-30
    choices:
      wandb: wandb
      eval: default
      train: baseline
      data: nsd
      paths: paths
      hydra/env: default
      hydra/callbacks: null
      hydra/job_logging: default
      hydra/hydra_logging: default
      hydra/hydra_help: default
      hydra/help: default
      hydra/sweeper: basic
      hydra/launcher: basic
      hydra/output: default
  verbose: false

```

# .dvc/cache/files/md5/b1/14b4dbddd4b453946cb1a555809659.dir

```dir
[{"md5": "47c9fb596d852b19d0a97abc4504a6ae", "relpath": "subj01_roi.npy"}]
```

# .dvc/cache/files/md5/bf/c2985bdddc338a605bfd927bf16943

```
[2025-09-29 00:23:45,494][__main__][INFO] - Config:
paths:
  data_dir: ${oc.env:DATA_DIR, ./data}
  raw_dir: ${paths.data_dir}/raw
  proc_dir: ${paths.data_dir}/processed
  artifacts_dir: ${paths.data_dir}/artifacts
data:
  name: nsd
  subjects:
  - subj01
  split:
    train_ratio: 0.9
    seed: 42
  paths:
    images_root: ${paths.raw_dir}/nsd/images
    fmri_root: ${paths.raw_dir}/nsd/fmri
    captions: ${paths.raw_dir}/nsd/captions.csv
  fmriprep:
    bids_root: ${paths.raw_dir}/nsd/bids
    out_root: ${paths.proc_dir}/nsd/fmriprep
    fs_license: ${oc.env:FS_LICENSE, ./licenses/freesurfer_license.txt}
  roi:
    mode: atlas
    atlas: glasser
    vector_dim: 2048
    out_dir: ${paths.proc_dir}/nsd/roi
train:
  max_epochs: 1
  batch_size: 2
  num_workers: 4
  precision: 32
  optimizer:
    lr: 0.001
  model:
    fmri_input_dim: 2048
    latent_dim: 768
    hidden:
    - 2048
    - 1024
    - 768
eval:
  metrics:
  - ssim
  - psnr
  - clip_score
wandb:
  enabled: true
  project: ${oc.env:WANDB_PROJECT, fmri2image}
  entity: ${oc.env:WANDB_ENTITY, null}
  mode: online
seed: 1337
device: cuda
run:
  name: baseline_train
  output_dir: outputs/${now:%Y-%m-%d}/${run.name}


```

# .dvc/cache/files/md5/d7/51713988987e9331980363e24189ce.dir

```dir
[]
```

# .dvc/cache/files/md5/f3/af829ab32b932a4b609de1671d9486.dir

```dir
[{"md5": "3e02189797c8eb27e07307bbd93b4667", "relpath": "2025-09-29/00-23-45/.hydra/config.yaml"}, {"md5": "6081bbb2697282d0d75d61047f8b0246", "relpath": "2025-09-29/00-23-45/.hydra/hydra.yaml"}, {"md5": "73bc718bee9b637ae12f8173a862007a", "relpath": "2025-09-29/00-23-45/.hydra/overrides.yaml"}, {"md5": "bfc2985bdddc338a605bfd927bf16943", "relpath": "2025-09-29/00-23-45/cli.log"}]
```

# .dvc/cache/files/md5/f8/3fdd2b68250633178df51bd1188267

```
[2025-09-28 23:32:44,175][__main__][INFO] - Config:
paths:
  data_dir: ${oc.env:DATA_DIR, ./data}
  raw_dir: ${paths.data_dir}/raw
  proc_dir: ${paths.data_dir}/processed
  artifacts_dir: ${paths.data_dir}/artifacts
data:
  name: nsd
  subjects:
  - subj01
  split:
    train_ratio: 0.9
    seed: 42
  paths:
    images_root: ${paths.raw_dir}/nsd/images
    fmri_root: ${paths.raw_dir}/nsd/fmri
    captions: ${paths.raw_dir}/nsd/captions.csv
train:
  max_epochs: 1
  batch_size: 2
  num_workers: 4
  precision: 32
  optimizer:
    lr: 0.001
  model:
    fmri_input_dim: 2048
    latent_dim: 768
    hidden:
    - 2048
    - 1024
    - 768
eval:
  metrics:
  - ssim
  - psnr
  - clip_score
wandb:
  enabled: true
  project: ${oc.env:WANDB_PROJECT, fmri2image}
  entity: ${oc.env:WANDB_ENTITY, null}
  mode: online
seed: 1337
device: cuda
run:
  name: baseline_train
  output_dir: outputs/${now:%Y-%m-%d}/${run.name}


```

# .dvc/cache/runs/a0/a08062eee6b20d8670490c4e18ed5bd7c398e54b56dd7a2a53fe69e8279d9a81/07077eba7949c9b84655401e95066d58d2ef2687e2a65213a2bafba4d8e3af6d

```
cmd: python -m fmri2image.cli ++run.name=baseline_train train=baseline
deps:
- path: configs/config.yaml
  hash: md5
  md5: c341062c96b325785ce81f82fb812de0
  size: 211
- path: configs/train/baseline.yaml
  hash: md5
  md5: 9801b9aeb6861586cf203650fc772a9e
  size: 231
- path: src/fmri2image/data/nsd_reader.py
  hash: md5
  md5: 084b9346dbc48b5ea6774ba4df7d14b9
  size: 726
- path: src/fmri2image/models/encoders/mlp_encoder.py
  hash: md5
  md5: 66e2f739ca3c679e83a79332265da2fa
  size: 448
- path: src/fmri2image/pipelines/baseline_train.py
  hash: md5
  md5: 83027afa14883e30166df53415571080
  size: 1646
outs:
- path: outputs
  hash: md5
  md5: 37f1c8f38e0280f7823b6dcf612120e5.dir
  size: 5310
  nfiles: 4

```

# .dvc/cache/runs/cd/cd509a493510f3503f9a440b74f8718a8bea0a51008a7653ca925ccdaab53bad/2d4c1e68afe6afc2c43867d8f6d0c7acad29633a7c955c8001f7350d3514e445

```
cmd: python -m fmri2image.cli ++run.name=baseline_train train=baseline
deps:
- path: configs/config.yaml
  hash: md5
  md5: c341062c96b325785ce81f82fb812de0
  size: 211
- path: configs/train/baseline.yaml
  hash: md5
  md5: 9801b9aeb6861586cf203650fc772a9e
  size: 231
- path: data/processed/nsd/roi/subj01_roi.npy
  hash: md5
  md5: 47c9fb596d852b19d0a97abc4504a6ae
  size: 983168
- path: src/fmri2image/data/nsd_reader.py
  hash: md5
  md5: 36d85d4ecab163fa022f4e5388d56dda
  size: 1110
- path: src/fmri2image/models/encoders/mlp_encoder.py
  hash: md5
  md5: 66e2f739ca3c679e83a79332265da2fa
  size: 448
- path: src/fmri2image/pipelines/baseline_train.py
  hash: md5
  md5: b00bd1875f518ced43fed7d8f9371021
  size: 1784
outs:
- path: outputs
  hash: md5
  md5: 6667798a2b9efe620802db99d520b5bf.dir
  size: 5854
  nfiles: 4

```

# .dvc/cache/runs/da/dadd61ca1f656a8e241bf6c07f1b22b767071f5d730b0615365db0b5e54b92e3/92e38a87ba248726bdb05bed8686c52e5f9862ed09ce9e2ae063eabeb28a97ab

```
cmd: python -m fmri2image.cli ++run.name=baseline_train train=baseline
deps:
- path: configs/config.yaml
  hash: md5
  md5: c341062c96b325785ce81f82fb812de0
  size: 211
- path: configs/train/baseline.yaml
  hash: md5
  md5: 9801b9aeb6861586cf203650fc772a9e
  size: 231
- path: data/processed/nsd/clip_text.npy
  hash: md5
  md5: 2577ce444d57eb6c9b162b4a4c8b09f4
  size: 6272
- path: data/processed/nsd/roi/subj01_roi.npy
  hash: md5
  md5: 47c9fb596d852b19d0a97abc4504a6ae
  size: 983168
- path: src/fmri2image/data/nsd_reader.py
  hash: md5
  md5: 36d85d4ecab163fa022f4e5388d56dda
  size: 1110
- path: src/fmri2image/models/encoders/mlp_encoder.py
  hash: md5
  md5: 66e2f739ca3c679e83a79332265da2fa
  size: 448
- path: src/fmri2image/pipelines/baseline_train.py
  hash: md5
  md5: 960ca40fa77c97e107130f93cb1fe5a5
  size: 2646
outs:
- path: outputs
  hash: md5
  md5: f3af829ab32b932a4b609de1671d9486.dir
  size: 5854
  nfiles: 4

```

# .dvc/config

```
[core]
    remote = localstore
['remote "localstore"']
    url = storage

```

# .dvc/storage/files/md5/d7/51713988987e9331980363e24189ce.dir

```dir
[]
```

# .dvcignore

```
# Add patterns of files dvc should ignore, which could improve
# the performance. Learn more at
# https://dvc.org/doc/user-guide/dvcignore

```

# .github/workflows/ci.yml

```yml
name: CI

on:
  push: { branches: ["main"] }
  pull_request:

jobs:
  lint-test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.10" }
      - name: Install
        run: |
          python -m pip install -U pip
          pip install -e .
          pip install pre-commit
          pre-commit install
      - name: Ruff & mypy
        run: |
          ruff check .
          mypy src
      - name: Tests
        run: pytest -q

```

# .github/workflows/docker-release.yml

```yml
name: Docker Release

on:
  workflow_dispatch:
  push:
    tags: ["v*.*.*"]

jobs:
  build-and-push:
    runs-on: ubuntu-latest
    permissions:
      contents: read
      packages: write
    steps:
      - uses: actions/checkout@v4
      - name: Login GHCR
        uses: docker/login-action@v3
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}
      - name: Build & Push
        uses: docker/build-push-action@v6
        with:
          push: true
          tags: ghcr.io/${{ github.repository }}:latest

```

# .gitignore

```
.venv/
__pycache__/
*.pyc
.env

# Data files
data/*
!data/*.dvc
!data/**/*.dvc
!data/.gitkeep
!data/README.md

.dvc/tmp/
.dvc/cache/
wandb/
outputs/
dist/

# allow Hydra configs
!configs/
!configs/**
/outputs

```

# .pre-commit-config.yaml

```yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.6.8
    hooks:
      - id: ruff
        args: [--fix]
  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.11.2
    hooks:
      - id: mypy
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.6.0
    hooks:
      - id: end-of-file-fixer
      - id: trailing-whitespace

```

# .pytest_cache/.gitignore

```
# Created by pytest automatically.
*

```

# .pytest_cache/CACHEDIR.TAG

```TAG
Signature: 8a477f597d28d172789f06886806bc55
# This file is a cache directory tag created by pytest.
# For information about cache directory tags, see:
#	https://bford.info/cachedir/spec.html

```

# .pytest_cache/README.md

```md
# pytest cache directory #

This directory contains data from the pytest's cache plugin,
which provides the `--lf` and `--ff` options, as well as the `cache` fixture.

**Do not** commit this to version control.

See [the docs](https://docs.pytest.org/en/stable/how-to/cache.html) for more information.

```

# .pytest_cache/v/cache/lastfailed

```
{}
```

# .pytest_cache/v/cache/nodeids

```
[
  "tests/test_smoke.py::test_cli_smoke"
]
```

# .pytest_cache/v/cache/stepwise

```
[]
```

# configs/config.yaml

```yaml
defaults:
  - paths: paths
  - data: nsd
  - train: baseline
  - eval: default
  - wandb: wandb
  - _self_

seed: 1337
device: cuda

run:
  name: baseline_debug
  output_dir: outputs/${now:%Y-%m-%d}/${run.name}

```

# configs/data/nsd_synth.yaml

```yaml
name: nsd_synth
paths:
  images_root: ${paths.raw_dir}/nsd_synth/images
  fmri_root: ${paths.raw_dir}/nsd_synth/fmri

```

# configs/data/nsd.yaml

```yaml
name: nsd
# Pointers; pentru început folosim mock-uri / fișiere mici
subjects: [subj01]
split:
  train_ratio: 0.9
  seed: 42
paths:
  images_root: ${paths.raw_dir}/nsd/images
  fmri_root: ${paths.raw_dir}/nsd/fmri
  captions: ${paths.raw_dir}/nsd/captions.csv

fmriprep:
  bids_root: ${paths.raw_dir}/nsd/bids
  out_root: ${paths.proc_dir}/nsd/fmriprep
  fs_license: ${oc.env:FS_LICENSE, ./licenses/freesurfer_license.txt}

roi:
  mode: "atlas" # "atlas" | "mask" | "whole_brain"
  atlas: "glasser" # ex: glasser / wang / schaefer
  vector_dim: 2048
  out_dir: ${paths.proc_dir}/nsd/roi

```

# configs/eval/default.yaml

```yaml
metrics: [ssim, psnr, clip_score]

```

# configs/paths/paths.yaml

```yaml
data_dir: ${oc.env:DATA_DIR, ./data}
raw_dir: ${paths.data_dir}/raw
proc_dir: ${paths.data_dir}/processed
artifacts_dir: ${paths.data_dir}/artifacts

```

# configs/train/baseline.yaml

```yaml
max_epochs: 1
batch_size: 2
num_workers: 4
precision: 32
optimizer:
  lr: 1e-3
model:
  fmri_input_dim: 2048 # placeholder ROI vectorizat
  latent_dim: 768 # țintă CLIP text/image latent (placeholder)
  hidden: [2048, 1024, 768]
loss:
  type: "contrastive" # "contrastive" | "cosine"
  temperature_init: 0.07 # initial temperature for InfoNCE (CLIP-style)
  symmetric: true # use both fMRI->text and text->fMRI
eval:
  topk: [1, 5]

```

# configs/train/debug.yaml

```yaml
max_epochs: 1
batch_size: 1
num_workers: 0
precision: 32

```

# configs/wandb.yaml

```yaml
enabled: true
project: ${oc.env:WANDB_PROJECT, fmri2image}
entity: ${oc.env:WANDB_ENTITY, null}
mode: online

```

# configs/wandb/wandb.yaml

```yaml
enabled: true
project: ${oc.env:WANDB_PROJECT, fmri2image}
entity: ${oc.env:WANDB_ENTITY, null}
mode: online

```

# data/processed/.keep

```

```

# data/processed/nsd/clip_text.npy

This is a binary file of the type: Binary

# data/processed/nsd/roi/subj01_roi.npy

This is a binary file of the type: Binary

# data/raw/nsd/captions.csv

```csv
image_id,caption
0,a dog on grass
1,a red car
2,a mountain scene

```

# data/raw/nsd/README.txt

```txt
NSD placeholder. Pentru date reale: configurați AWS CLI și sincronizați din bucketul NSD.
Struc.: raw/nsd/images/, raw/nsd/fmri/, raw/nsd/captions.csv

```

# docker-compose.yml

```yml
services:
  dev:
    build: .
    image: fmri2image:dev
    working_dir: /workspace
    volumes:
      - ./:/workspace
    environment:
      - WANDB_PROJECT
      - WANDB_ENTITY
    tty: true
    deploy:
      resources:
        reservations:
          devices:
            - capabilities: [gpu]

```

# Dockerfile

```
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget curl ca-certificates python3 python3-pip python3-venv ffmpeg && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
COPY pyproject.toml README.md ./
RUN pip3 install -U pip && pip3 install -e .

# Optional: preinstall DVC S3 deps
RUN pip3 install "dvc[s3]" pre-commit

ENV PYTHONPATH=/workspace/src
COPY . .
CMD ["bash"]

```

# dvc.lock

```lock
schema: '2.0'
stages:
  prepare_data:
    cmd: python -m fmri2image.data.download_nsd --raw_root data/raw
    deps:
    - path: configs/paths/paths.yaml
      hash: md5
      md5: 8f28bb84db3bdc7a4e9ff462057c7078
      size: 149
    - path: src/fmri2image/data/download_nsd.py
      hash: md5
      md5: abd4099492831141fa689f051c584af9
      size: 920
    outs:
    - path: data/raw/nsd
      hash: md5
      md5: 79a62e2ffa10ebce2562583f42158d68.dir
      size: 219
      nfiles: 2
  baseline_train:
    cmd: python -m fmri2image.cli ++run.name=baseline_train train=baseline
    deps:
    - path: configs/config.yaml
      hash: md5
      md5: c341062c96b325785ce81f82fb812de0
      size: 211
    - path: configs/train/baseline.yaml
      hash: md5
      md5: 9801b9aeb6861586cf203650fc772a9e
      size: 231
    - path: data/processed/nsd/clip_text.npy
      hash: md5
      md5: 2577ce444d57eb6c9b162b4a4c8b09f4
      size: 6272
    - path: data/processed/nsd/roi/subj01_roi.npy
      hash: md5
      md5: 47c9fb596d852b19d0a97abc4504a6ae
      size: 983168
    - path: src/fmri2image/data/nsd_reader.py
      hash: md5
      md5: 36d85d4ecab163fa022f4e5388d56dda
      size: 1110
    - path: src/fmri2image/models/encoders/mlp_encoder.py
      hash: md5
      md5: 66e2f739ca3c679e83a79332265da2fa
      size: 448
    - path: src/fmri2image/pipelines/baseline_train.py
      hash: md5
      md5: 960ca40fa77c97e107130f93cb1fe5a5
      size: 2646
    outs:
    - path: outputs
      hash: md5
      md5: f3af829ab32b932a4b609de1671d9486.dir
      size: 5854
      nfiles: 4
  roi_extract:
    cmd: "python -m fmri2image.data.extract_roi --fmriprep_dir data/processed/nsd/fmriprep
      --out_dir data/processed/nsd/roi --mode atlas --atlas glasser --vector_dim 2048
      --subject subj01\n"
    deps:
    - path: configs/data/nsd.yaml
      hash: md5
      md5: aa0676ebabf3bfd4721381d7f051c949
      size: 589
    - path: src/fmri2image/data/extract_roi.py
      hash: md5
      md5: 972f9e67547e89b6abb352e69d6ff970
      size: 1133
    outs:
    - path: data/processed/nsd/roi
      hash: md5
      md5: b114b4dbddd4b453946cb1a555809659.dir
      size: 983168
      nfiles: 1
  clip_text:
    cmd: "python -m fmri2image.text.clip_text --captions data/raw/nsd/captions.csv
      --out data/processed/nsd/clip_text.npy\n"
    deps:
    - path: data/raw/nsd/captions.csv
      hash: md5
      md5: 2cfdc695fc4d02a0aef4e9c1185bf1ae
      size: 65
    - path: src/fmri2image/text/clip_text.py
      hash: md5
      md5: 6a157c4ab11fcd3922ff2414e344cc67
      size: 1277
    outs:
    - path: data/processed/nsd/clip_text.npy
      hash: md5
      md5: 2577ce444d57eb6c9b162b4a4c8b09f4
      size: 6272

```

# dvc.yaml

```yaml
stages:
  prepare_data:
    cmd: python -m fmri2image.data.download_nsd --raw_root data/raw
    deps:
      - src/fmri2image/data/download_nsd.py
      - configs/paths/paths.yaml
    outs:
      - data/raw/nsd:
          persist: true

  roi_extract:
    cmd: >
      python -m fmri2image.data.extract_roi
      --fmriprep_dir data/processed/nsd/fmriprep
      --out_dir data/processed/nsd/roi
      --mode atlas
      --atlas glasser
      --vector_dim 2048
      --subject subj01
    deps:
      - src/fmri2image/data/extract_roi.py
      - configs/data/nsd.yaml
    outs:
      - data/processed/nsd/roi:
          persist: true

  clip_text:
    cmd: >
      python -m fmri2image.text.clip_text
      --captions data/raw/nsd/captions.csv
      --out data/processed/nsd/clip_text.npy
    deps:
      - src/fmri2image/text/clip_text.py
      - data/raw/nsd/captions.csv
    outs:
      - data/processed/nsd/clip_text.npy:
          persist: true

  baseline_train:
    cmd: python -m fmri2image.cli ++run.name=baseline_train train=baseline
    deps:
      - src/fmri2image/pipelines/baseline_train.py
      - src/fmri2image/models/encoders/mlp_encoder.py
      - src/fmri2image/data/nsd_reader.py
      - configs/train/baseline.yaml
      - configs/config.yaml
      - data/processed/nsd/roi/subj01_roi.npy
      - data/processed/nsd/clip_text.npy
    outs:
      - outputs

```

# Makefile

```
PY=python

install:
	python -m venv .venv && . .venv/bin/activate && pip install -U pip && pip install -e . && pre-commit install

lint:
	ruff check .
	mypy src

test:
	pytest -q

run:
	$(PY) -m fmri2image.cli

dvc-init:
	dvc init -f
	dvc remote add -d origin $${DVC_REMOTE}
	dvc gc -w

docker-build:
	docker build -t fmri2image:dev .

docker-run:
	docker run --rm -it -v $$(pwd):/workspace -e WANDB_PROJECT -e WANDB_ENTITY fmri2image:dev

data-mock:
	python -m fmri2image.data.download_nsd --raw_root $(DATA_DIR)/raw

dvc-track:
	dvc add data/raw && git add data/raw.dvc && git commit -m "data: update raw" || true

```

# outputs/2025-09-29/22-20-55/.hydra/config.yaml

```yaml
paths:
  data_dir: ${oc.env:DATA_DIR, ./data}
  raw_dir: ${paths.data_dir}/raw
  proc_dir: ${paths.data_dir}/processed
  artifacts_dir: ${paths.data_dir}/artifacts
data:
  name: nsd
  subjects:
  - subj01
  split:
    train_ratio: 0.9
    seed: 42
  paths:
    images_root: ${paths.raw_dir}/nsd/images
    fmri_root: ${paths.raw_dir}/nsd/fmri
    captions: ${paths.raw_dir}/nsd/captions.csv
  fmriprep:
    bids_root: ${paths.raw_dir}/nsd/bids
    out_root: ${paths.proc_dir}/nsd/fmriprep
    fs_license: ${oc.env:FS_LICENSE, ./licenses/freesurfer_license.txt}
  roi:
    mode: atlas
    atlas: glasser
    vector_dim: 2048
    out_dir: ${paths.proc_dir}/nsd/roi
train:
  max_epochs: 1
  batch_size: 2
  num_workers: 4
  precision: 32
  optimizer:
    lr: 0.001
  model:
    fmri_input_dim: 2048
    latent_dim: 768
    hidden:
    - 2048
    - 1024
    - 768
  loss:
    type: contrastive
    temperature_init: 0.07
    symmetric: true
  eval:
    topk:
    - 1
    - 5
eval:
  metrics:
  - ssim
  - psnr
  - clip_score
wandb:
  enabled: true
  project: ${oc.env:WANDB_PROJECT, fmri2image}
  entity: ${oc.env:WANDB_ENTITY, null}
  mode: online
seed: 1337
device: cuda
run:
  name: baseline_train
  output_dir: outputs/${now:%Y-%m-%d}/${run.name}

```

# outputs/2025-09-29/22-20-55/.hydra/hydra.yaml

```yaml
hydra:
  run:
    dir: outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}
  sweep:
    dir: multirun/${now:%Y-%m-%d}/${now:%H-%M-%S}
    subdir: ${hydra.job.num}
  launcher:
    _target_: hydra._internal.core_plugins.basic_launcher.BasicLauncher
  sweeper:
    _target_: hydra._internal.core_plugins.basic_sweeper.BasicSweeper
    max_batch_size: null
    params: null
  help:
    app_name: ${hydra.job.name}
    header: '${hydra.help.app_name} is powered by Hydra.

      '
    footer: 'Powered by Hydra (https://hydra.cc)

      Use --hydra-help to view Hydra specific help

      '
    template: '${hydra.help.header}

      == Configuration groups ==

      Compose your configuration from those groups (group=option)


      $APP_CONFIG_GROUPS


      == Config ==

      Override anything in the config (foo.bar=value)


      $CONFIG


      ${hydra.help.footer}

      '
  hydra_help:
    template: 'Hydra (${hydra.runtime.version})

      See https://hydra.cc for more info.


      == Flags ==

      $FLAGS_HELP


      == Configuration groups ==

      Compose your configuration from those groups (For example, append hydra/job_logging=disabled
      to command line)


      $HYDRA_CONFIG_GROUPS


      Use ''--cfg hydra'' to Show the Hydra config.

      '
    hydra_help: ???
  hydra_logging:
    version: 1
    formatters:
      simple:
        format: '[%(asctime)s][HYDRA] %(message)s'
    handlers:
      console:
        class: logging.StreamHandler
        formatter: simple
        stream: ext://sys.stdout
    root:
      level: INFO
      handlers:
      - console
    loggers:
      logging_example:
        level: DEBUG
    disable_existing_loggers: false
  job_logging:
    version: 1
    formatters:
      simple:
        format: '[%(asctime)s][%(name)s][%(levelname)s] - %(message)s'
    handlers:
      console:
        class: logging.StreamHandler
        formatter: simple
        stream: ext://sys.stdout
      file:
        class: logging.FileHandler
        formatter: simple
        filename: ${hydra.runtime.output_dir}/${hydra.job.name}.log
    root:
      level: INFO
      handlers:
      - console
      - file
    disable_existing_loggers: false
  env: {}
  mode: RUN
  searchpath: []
  callbacks: {}
  output_subdir: .hydra
  overrides:
    hydra:
    - hydra.mode=RUN
    task:
    - ++run.name=baseline_train
    - train=baseline
  job:
    name: cli
    chdir: null
    override_dirname: ++run.name=baseline_train,train=baseline
    id: ???
    num: ???
    config_name: config
    env_set: {}
    env_copy: []
    config:
      override_dirname:
        kv_sep: '='
        item_sep: ','
        exclude_keys: []
  runtime:
    version: 1.3.2
    version_base: '1.3'
    cwd: /home/tonystark/Desktop/Bachelor/fmri2image
    config_sources:
    - path: hydra.conf
      schema: pkg
      provider: hydra
    - path: /home/tonystark/Desktop/Bachelor/fmri2image/configs
      schema: file
      provider: main
    - path: ''
      schema: structured
      provider: schema
    output_dir: /home/tonystark/Desktop/Bachelor/fmri2image/outputs/2025-09-29/22-20-55
    choices:
      wandb: wandb
      eval: default
      train: baseline
      data: nsd
      paths: paths
      hydra/env: default
      hydra/callbacks: null
      hydra/job_logging: default
      hydra/hydra_logging: default
      hydra/hydra_help: default
      hydra/help: default
      hydra/sweeper: basic
      hydra/launcher: basic
      hydra/output: default
  verbose: false

```

# outputs/2025-09-29/22-20-55/.hydra/overrides.yaml

```yaml
- ++run.name=baseline_train
- train=baseline

```

# outputs/2025-09-29/22-20-55/cli.log

```log
[2025-09-29 22:20:55,427][__main__][INFO] - Config:
paths:
  data_dir: ${oc.env:DATA_DIR, ./data}
  raw_dir: ${paths.data_dir}/raw
  proc_dir: ${paths.data_dir}/processed
  artifacts_dir: ${paths.data_dir}/artifacts
data:
  name: nsd
  subjects:
  - subj01
  split:
    train_ratio: 0.9
    seed: 42
  paths:
    images_root: ${paths.raw_dir}/nsd/images
    fmri_root: ${paths.raw_dir}/nsd/fmri
    captions: ${paths.raw_dir}/nsd/captions.csv
  fmriprep:
    bids_root: ${paths.raw_dir}/nsd/bids
    out_root: ${paths.proc_dir}/nsd/fmriprep
    fs_license: ${oc.env:FS_LICENSE, ./licenses/freesurfer_license.txt}
  roi:
    mode: atlas
    atlas: glasser
    vector_dim: 2048
    out_dir: ${paths.proc_dir}/nsd/roi
train:
  max_epochs: 1
  batch_size: 2
  num_workers: 4
  precision: 32
  optimizer:
    lr: 0.001
  model:
    fmri_input_dim: 2048
    latent_dim: 768
    hidden:
    - 2048
    - 1024
    - 768
  loss:
    type: contrastive
    temperature_init: 0.07
    symmetric: true
  eval:
    topk:
    - 1
    - 5
eval:
  metrics:
  - ssim
  - psnr
  - clip_score
wandb:
  enabled: true
  project: ${oc.env:WANDB_PROJECT, fmri2image}
  entity: ${oc.env:WANDB_ENTITY, null}
  mode: online
seed: 1337
device: cuda
run:
  name: baseline_train
  output_dir: outputs/${now:%Y-%m-%d}/${run.name}


```

# pyproject.toml

```toml
[build-system]
requires = ["setuptools>=69", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "fmri2image"
version = "0.1.0"
description = "fMRI→Image reconstruction pipeline (Hydra, DVC, Lightning, W&B)"
readme = "README.md"
requires-python = ">=3.10"
dependencies = [
  "torch>=2.3",
  "pytorch-lightning>=2.4",
  "hydra-core>=1.3",
  "omegaconf>=2.3",
  "dvc[s3]>=3.50",
  "numpy>=1.26",
  "pandas>=2.2",
  "scikit-learn>=1.5",
  "einops>=0.8",
  "tqdm>=4.66",
  "wandb>=0.17",
  "rich>=13.7",
  "pytest>=8.3",
  "pytest-cov>=5.0",
  "mypy>=1.11",
  "ruff>=0.6",
  "xarray>=2024.6",
  "nibabel>=5.2",
  "nilearn>=0.10",
  "h5py>=3.11",
  "open_clip_torch>=2.24.0",
]

[tool.ruff]
line-length = 100
select = ["E","F","I","B","UP"]
exclude = ["data","build",".venv"]

[tool.mypy]
python_version = "3.10"
ignore_missing_imports = true
strict = false

[tool.setuptools]
package-dir = {"" = "src"}

[tool.setuptools.packages.find]
where = ["src"]

```

# README.md

```md
# fmri2image

Industrial-grade scaffold (Hydra + DVC + Docker + CI + tests) pentru reconstrucție fMRI→Imagine.

## Setup rapid

\`\`\`bash
python -m venv .venv && source .venv/bin/activate
pip install -U pip && pip install -e .
pre-commit install
cp .env.example .env
export DATA_DIR=$(pwd)/data
make dvc-init
pytest -q
python -m fmri2image.cli
\`\`\`

```

# src/fmri2image.egg-info/dependency_links.txt

```txt


```

# src/fmri2image.egg-info/PKG-INFO

```
Metadata-Version: 2.4
Name: fmri2image
Version: 0.1.0
Summary: fMRI→Image reconstruction pipeline (Hydra, DVC, Lightning, W&B)
Requires-Python: >=3.10
Description-Content-Type: text/markdown
Requires-Dist: torch>=2.3
Requires-Dist: pytorch-lightning>=2.4
Requires-Dist: hydra-core>=1.3
Requires-Dist: omegaconf>=2.3
Requires-Dist: dvc[s3]>=3.50
Requires-Dist: numpy>=1.26
Requires-Dist: pandas>=2.2
Requires-Dist: scikit-learn>=1.5
Requires-Dist: einops>=0.8
Requires-Dist: tqdm>=4.66
Requires-Dist: wandb>=0.17
Requires-Dist: rich>=13.7
Requires-Dist: pytest>=8.3
Requires-Dist: pytest-cov>=5.0
Requires-Dist: mypy>=1.11
Requires-Dist: ruff>=0.6
Requires-Dist: xarray>=2024.6
Requires-Dist: nibabel>=5.2
Requires-Dist: nilearn>=0.10
Requires-Dist: h5py>=3.11
Requires-Dist: open_clip_torch>=2.24.0

# fmri2image

Industrial-grade scaffold (Hydra + DVC + Docker + CI + tests) pentru reconstrucție fMRI→Imagine.

## Setup rapid

\`\`\`bash
python -m venv .venv && source .venv/bin/activate
pip install -U pip && pip install -e .
pre-commit install
cp .env.example .env
export DATA_DIR=$(pwd)/data
make dvc-init
pytest -q
python -m fmri2image.cli
\`\`\`

```

# src/fmri2image.egg-info/requires.txt

```txt
torch>=2.3
pytorch-lightning>=2.4
hydra-core>=1.3
omegaconf>=2.3
dvc[s3]>=3.50
numpy>=1.26
pandas>=2.2
scikit-learn>=1.5
einops>=0.8
tqdm>=4.66
wandb>=0.17
rich>=13.7
pytest>=8.3
pytest-cov>=5.0
mypy>=1.11
ruff>=0.6
xarray>=2024.6
nibabel>=5.2
nilearn>=0.10
h5py>=3.11
open_clip_torch>=2.24.0

```

# src/fmri2image.egg-info/SOURCES.txt

```txt
README.md
pyproject.toml
src/fmri2image/__init__.py
src/fmri2image/cli.py
src/fmri2image.egg-info/PKG-INFO
src/fmri2image.egg-info/SOURCES.txt
src/fmri2image.egg-info/dependency_links.txt
src/fmri2image.egg-info/requires.txt
src/fmri2image.egg-info/top_level.txt
src/fmri2image/data/datamodule.py
src/fmri2image/data/download_nsd.py
src/fmri2image/data/extract_roi.py
src/fmri2image/data/nsd_reader.py
src/fmri2image/models/encoders/mlp_encoder.py
src/fmri2image/pipelines/baseline_train.py
src/fmri2image/pipelines/eval_metrics.py
src/fmri2image/utils/logging.py
src/fmri2image/utils/seed.py
tests/test_smoke.py
```

# src/fmri2image.egg-info/top_level.txt

```txt
fmri2image

```

# src/fmri2image/__init__.py

```py

```

# src/fmri2image/cli.py

```py
import os
import hydra
from omegaconf import DictConfig, OmegaConf
from .utils.logging import get_logger
from .utils.seed import seed_everything
from .pipelines.baseline_train import run_baseline

log = get_logger(__name__)

@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig):
    seed_everything(cfg.get("seed", 1337))
    log.info("Config:\n%s", OmegaConf.to_yaml(cfg))
    os.makedirs(cfg.run.output_dir, exist_ok=True)
    run_baseline(cfg)

if __name__ == "__main__":
    main()

```

# src/fmri2image/data/datamodule.py

```py
from torch.utils.data import Dataset, DataLoader
import torch

class FMRITextDataset(Dataset):
    def __init__(self, X, texts):
        self.X, self.texts = X, texts
    def __len__(self): 
        return len(self.texts)
    def __getitem__(self, idx):
        # return the fmri sample AND its index (for CLIP alignment)
        return torch.tensor(self.X[idx]), (torch.tensor(idx, dtype=torch.long), self.texts[idx])

def make_loaders(X, texts, batch_size=2, num_workers=0):
    ds = FMRITextDataset(X, texts)
    return DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)

```

# src/fmri2image/data/download_nsd.py

```py
from pathlib import Path
import argparse, json

def main(raw_root: str):
    root = Path(raw_root); (root/"nsd"/"images").mkdir(parents=True, exist_ok=True)
    (root/"nsd"/"fmri").mkdir(parents=True, exist_ok=True)
    # MOCK: scriem un captions CSV minim și un README cu pași reali
    captions = root/"nsd"/"captions.csv"
    if not captions.exists():
        captions.write_text("image_id,caption\n0,a dog on grass\n1,a red car\n2,a mountain scene\n")
    readme = root/"nsd"/"README.txt"
    readme.write_text(
        "NSD placeholder. Pentru date reale: configurați AWS CLI și sincronizați din bucketul NSD.\n"
        "Struc.: raw/nsd/images/, raw/nsd/fmri/, raw/nsd/captions.csv\n"
    )
    print(f"[ok] NSD mock at: {root/'nsd'}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_root", type=str, required=True)
    args = ap.parse_args()
    main(args.raw_root)

```

# src/fmri2image/data/extract_roi.py

```py
from pathlib import Path
import argparse
import numpy as np

def main(fmriprep_dir: str, out_dir: str, mode: str, atlas: str, vec_dim: int, subject: str = "subj01"):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    # TODO (real): load preprocessed BOLD + apply ROI parcellation (nilearn + atlas)
    # MOCK: generate a deterministic tensor so runs are reproducible
    rng = np.random.default_rng(1337)
    roi = rng.standard_normal((120, vec_dim), dtype=np.float32)  # 120 "samples" x vec_dim
    np.save(out / f"{subject}_roi.npy", roi)
    print(f"[ok] ROI tensor mock -> {out / f'{subject}_roi.npy'}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fmriprep_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--mode", default="atlas")
    ap.add_argument("--atlas", default="glasser")
    ap.add_argument("--vector_dim", type=int, default=2048)
    ap.add_argument("--subject", type=str, default="subj01")
    args = ap.parse_args()
    main(args.fmriprep_dir, args.out_dir, args.mode, args.atlas, args.vector_dim, args.subject)

```

# src/fmri2image/data/nsd_reader.py

```py
from pathlib import Path
import pandas as pd
import numpy as np

class NSDReader:
    def __init__(self, images_root: str, fmri_root: str, captions: str, roi_dir: str | None = None, subject: str = "subj01"):
        self.images_root = Path(images_root)
        self.fmri_root = Path(fmri_root)
        self.captions = Path(captions)
        self.roi_dir = Path(roi_dir) if roi_dir else None
        self.subject = subject

    def load(self, n: int | None = None, fmri_dim: int = 2048, seed: int = 1337):
        df = pd.read_csv(self.captions)
        if n is not None:
            df = df.head(n)
        texts = df["caption"].tolist()

        if self.roi_dir is not None:
            roi_file = self.roi_dir / f"{self.subject}_roi.npy"
            if roi_file.exists():
                X = np.load(roi_file)
                if n is not None:
                    X = X[:n]
                return X.astype("float32"), texts

        # Fallback mock if no ROI file exists
        rng = np.random.default_rng(seed)
        X = rng.standard_normal((len(df), fmri_dim), dtype=np.float32)
        return X, texts

```

# src/fmri2image/models/encoders/mlp_encoder.py

```py
import torch, torch.nn as nn

class FMRIEncoderMLP(nn.Module):
    def __init__(self, in_dim: int, latent_dim: int, hidden: list[int]):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers += [nn.Linear(prev, latent_dim)]
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)

```

# src/fmri2image/pipelines/baseline_train.py

```py
from omegaconf import DictConfig
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl

from ..models.encoders.mlp_encoder import FMRIEncoderMLP
from ..data.nsd_reader import NSDReader
from ..data.datamodule import make_loaders
from .metrics import topk_retrieval


class ClipStyleContrastiveLoss(nn.Module):
    """
    CLIP-style InfoNCE with learnable temperature (logit scale).
    Uses in-batch negatives. Optionally symmetric (z->t and t->z).
    """
    def __init__(self, temperature_init: float = 0.07, symmetric: bool = True):
        super().__init__()
        # CLIP uses a learnable logit scale initialized to log(1/temperature)
        self.logit_scale = nn.Parameter(
            torch.tensor(np.log(1.0 / float(temperature_init)), dtype=torch.float32)
        )
        self.symmetric = bool(symmetric)

    def forward(self, z: torch.Tensor, t: torch.Tensor) -> dict:
        """
        Args:
            z: [B, D] fMRI->latent predictions
            t: [B, D] normalized CLIP text embeddings
        Returns:
            dict(loss=..., logits_zt=..., logits_tz|None=..., temp=...)
        """
        # Normalize both (t should already be normalized, but we re-normalize safely)
        z = z / (z.norm(dim=-1, keepdim=True) + 1e-8)
        t = t / (t.norm(dim=-1, keepdim=True) + 1e-8)

        # temperature = 1 / softmax temperature; we multiply similarities by exp(logit_scale)
        logit_scale = self.logit_scale.clamp(-5.0, 8.0)  # clamp for numerical stability
        scale = torch.exp(logit_scale)

        # Similarities (cosine since vectors are unit norm) scaled by temperature
        logits_zt = (z @ t.t()) * scale  # [B, B]
        target = torch.arange(z.size(0), device=z.device)
        loss_zt = nn.functional.cross_entropy(logits_zt, target)

        if self.symmetric:
            logits_tz = (t @ z.t()) * scale
            loss_tz = nn.functional.cross_entropy(logits_tz, target)
            loss = 0.5 * (loss_zt + loss_tz)
        else:
            logits_tz = None
            loss = loss_zt

        # Report the *temperature* for logging (inverse of scale)
        temp = torch.exp(-logit_scale)
        return {"loss": loss, "logits_zt": logits_zt, "logits_tz": logits_tz, "temp": temp}


class LitModule(pl.LightningModule):
    def __init__(self, cfg: DictConfig, clip_text_feats: np.ndarray):
        super().__init__()
        self.cfg = cfg
        m = cfg.train.model

        # Project fMRI -> CLIP text embedding dimension
        out_dim = int(clip_text_feats.shape[1])
        self.encoder = FMRIEncoderMLP(m.fmri_input_dim, out_dim, m.hidden)

        self.criterion = ClipStyleContrastiveLoss(
            temperature_init=float(cfg.train.loss.temperature_init),
            symmetric=bool(cfg.train.loss.get("symmetric", True)),
        )

        # Register CLIP text features as a non-persistent buffer (moved with device)
        self.register_buffer(
            "clip_text_feats",
            torch.tensor(clip_text_feats, dtype=torch.float32),
            persistent=False,
        )

        self.topk = tuple(getattr(getattr(cfg.train, "eval", {}), "topk", [1, 5]))
        # Save hyperparameters except the big array (stored as buffer)
        try:
            self.save_hyperparameters({"cfg": cfg})
        except Exception:
            pass

    def training_step(self, batch, _):
        x, (idx, _texts) = batch                 # x: [B, in_dim], idx: [B]
        z = self.encoder(x)                      # [B, D]
        t = self.clip_text_feats.index_select(0, idx.long())  # [B, D] on correct device

        out = self.criterion(z, t)
        self.log("train/loss", out["loss"], prog_bar=True, on_step=True, on_epoch=True)
        self.log("train/temp", out["temp"], prog_bar=False, on_step=True, on_epoch=True)

        # Retrieval metrics within-batch (ranking unaffected by temperature)
        with torch.no_grad():
            sim_zt = out["logits_zt"] / torch.exp(self.criterion.logit_scale)  # remove scale
            m_zt = topk_retrieval(sim_zt, self.topk)
            for k, v in m_zt.items():
                self.log(f"train/retrieval_zt_top{k}", v, prog_bar=True, on_step=False, on_epoch=True)

            if out["logits_tz"] is not None:
                sim_tz = out["logits_tz"] / torch.exp(self.criterion.logit_scale)
                m_tz = topk_retrieval(sim_tz, self.topk)
                for k, v in m_tz.items():
                    self.log(f"train/retrieval_tz_top{k}", v, prog_bar=False, on_step=False, on_epoch=True)

        return out["loss"]

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=float(self.cfg.train.optimizer.lr))


def run_baseline(cfg: DictConfig):
    # ---- Data ----
    reader = NSDReader(
        cfg.data.paths.images_root,
        cfg.data.paths.fmri_root,
        cfg.data.paths.captions,
        roi_dir=cfg.data.roi.out_dir,
        subject=cfg.data.subjects[0] if "subjects" in cfg.data and cfg.data.subjects else "subj01",
    )
    X, texts = reader.load(n=64, fmri_dim=cfg.train.model.fmri_input_dim)

    # Load CLIP text embeddings saved by DVC stage
    clip_feats = np.load("data/processed/nsd/clip_text.npy")
    # Keep arrays aligned
    n = min(len(X), len(clip_feats))
    X, texts, clip_feats = X[:n], texts[:n], clip_feats[:n]

    dl = make_loaders(X, texts, cfg.train.batch_size, cfg.train.num_workers)
    model = LitModule(cfg, clip_feats)

    # Optional: W&B logger (safe fallback to no-logger)
    logger = False
    try:
        if getattr(cfg.wandb, "enabled", False):
            from pytorch_lightning.loggers import WandbLogger
            logger = WandbLogger(
                project=getattr(cfg.wandb, "project", "fmri2image"),
                entity=getattr(cfg.wandb, "entity", None),
                log_model=False,
            )
    except Exception:
        logger = False

    trainer = pl.Trainer(
        max_epochs=cfg.train.max_epochs,
        precision=cfg.train.precision,
        default_root_dir=cfg.run.output_dir,
        enable_checkpointing=False,
        logger=logger,
    )
    trainer.fit(model, dl)

```

# src/fmri2image/pipelines/eval_metrics.py

```py
def ssim_placeholder(img_a, img_b): return 0.0
def psnr_placeholder(img_a, img_b): return 0.0
def clip_score_placeholder(img, text): return 0.0

```

# src/fmri2image/pipelines/metrics.py

```py
import torch
from typing import Sequence

@torch.no_grad()
def topk_retrieval(sim: torch.Tensor, ks: Sequence[int] = (1, 5)) -> dict:
    """
    sim: [B, B] similarity matrix (rows = queries, cols = candidates).
    Diagonal entries are positives.
    Returns: {'top1': ..., 'top5': ...} accuracies in [0,1].
    """
    B = sim.size(0)
    target = torch.arange(B, device=sim.device)
    ranks = (-sim).argsort(dim=1)  # descending similarity
    metrics = {}
    for k in ks:
        pred_ok = (ranks[:, :k] == target[:, None]).any(dim=1).float().mean().item()
        metrics[f"top{k}"] = pred_ok
    return metrics

```

# src/fmri2image/text/clip_text.py

```py
from pathlib import Path
import argparse
import torch
import numpy as np
import pandas as pd
import open_clip

@torch.no_grad()
def main(captions_csv: str, out_path: str,
         model_name: str = "ViT-B-32",
         pretrained: str = "laion2b_s34b_b79k",
         device: str = "cuda"):
    df = pd.read_csv(captions_csv)
    texts = df["caption"].astype(str).tolist()

    model, _, _ = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained, device=device
    )
    tokenizer = open_clip.get_tokenizer(model_name)
    toks = tokenizer(texts).to(device)
    feats = model.encode_text(toks)
    feats = feats / feats.norm(dim=-1, keepdim=True)
    out = Path(out_path); out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, feats.float().cpu().numpy())
    print(f"[ok] saved CLIP text features -> {out}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--captions", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model_name", default="ViT-B-32")
    ap.add_argument("--pretrained", default="laion2b_s34b_b79k")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    main(args.captions, args.out, args.model_name, args.pretrained, args.device)

```

# src/fmri2image/utils/logging.py

```py
from logging import getLogger, basicConfig, INFO, Formatter, StreamHandler
import sys

def get_logger(name: str):
    logger = getLogger(name)
    if not logger.handlers:
        handler = StreamHandler(sys.stdout)
        handler.setFormatter(Formatter("[%(levelname)s] %(name)s: %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(INFO)
    return logger

```

# src/fmri2image/utils/seed.py

```py
import random, numpy as np, torch

def seed_everything(seed: int = 1337):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

```

# tests/conftest.py

```py

```

# tests/test_smoke.py

```py
from subprocess import run, CalledProcessError
import sys

def test_cli_smoke():
    # Verifică dacă scriptul pornește și rulează un epoch rapid
    try:
        result = run([sys.executable, "-m", "fmri2image.cli"], check=True, capture_output=True)
    except CalledProcessError as e:
        print(e.stdout.decode(), e.stderr.decode())
        raise

```

