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

# .dvc/cache/files/md5/04/a629f13e039b482e0b6686f20ab5a3

This is a binary file of the type: Binary

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

# .dvc/cache/files/md5/6d/90c392139922ff1e0a10240d3b1786

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
    output_dir: /home/tonystark/Desktop/Bachelor/fmri2image/outputs/2025-09-30/00-21-16
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

# .dvc/cache/files/md5/7d/18581bb9713d9ce2e02effda2a1ded

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
    output_dir: /home/tonystark/Desktop/Bachelor/fmri2image/outputs/2025-09-29/22-39-01
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

# .dvc/cache/files/md5/33/4da91462da50992134edf89d1c7eaf

```
[2025-09-30 00:09:04,125][__main__][INFO] - Config:
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
    weights:
      contrastive: 1.0
      cca: 0.1
  eval:
    topk:
    - 1
    - 5
  encoder: mlp
  gnn:
    use_identity_adj: true
    dropout: 0.1
  vit3d:
    time_steps: 8
    patch: 1
    depth: 2
    heads: 4
    mlp_ratio: 2.0
    dropout: 0.1
  cca:
    enabled: true
    proj_dim: 128
  pretrained:
    path: data/artifacts/encoder_pretrained.ckpt
eval:
  metrics:
  - ssim
  - psnr
  - clip_score
wandb:
  enabled: false
  project: ${oc.env:WANDB_PROJECT, fmri2image}
  entity: ${oc.env:WANDB_ENTITY, null}
  mode: online
seed: 1337
device: cuda
run:
  name: baseline_train
  output_dir: outputs/${now:%Y-%m-%d}/${run.name}


```

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

# .dvc/cache/files/md5/69/759c9ee2ed9b9fb9ba32a4d660985b.dir

```dir
[{"md5": "cb2f2259e46435d7ee1a6000ddbfe736", "relpath": "2025-09-30/00-21-16/.hydra/config.yaml"}, {"md5": "6d90c392139922ff1e0a10240d3b1786", "relpath": "2025-09-30/00-21-16/.hydra/hydra.yaml"}, {"md5": "73bc718bee9b637ae12f8173a862007a", "relpath": "2025-09-30/00-21-16/.hydra/overrides.yaml"}, {"md5": "77abf48e4a307692ce4ee6bf530dacce", "relpath": "2025-09-30/00-21-16/cli.log"}]
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

# .dvc/cache/files/md5/72/7145e6dffb6fe0ee0aa77472970cc9

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
    output_dir: /home/tonystark/Desktop/Bachelor/fmri2image/outputs/2025-09-29/22-28-31
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

# .dvc/cache/files/md5/73/bc718bee9b637ae12f8173a862007a

```
- ++run.name=baseline_train
- train=baseline

```

# .dvc/cache/files/md5/77/abf48e4a307692ce4ee6bf530dacce

```
[2025-09-30 00:21:16,242][__main__][INFO] - Config:
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
    weights:
      contrastive: 1.0
      cca: 0.05
  eval:
    topk:
    - 1
    - 5
  encoder: mlp
  gnn:
    use_identity_adj: true
    dropout: 0.1
  vit3d:
    time_steps: 8
    patch: 1
    depth: 2
    heads: 4
    mlp_ratio: 2.0
    dropout: 0.1
  cca:
    enabled: true
    proj_dim: 128
  pretrained:
    path: data/artifacts/encoder_pretrained.ckpt
eval:
  metrics:
  - ssim
  - psnr
  - clip_score
wandb:
  enabled: false
  project: ${oc.env:WANDB_PROJECT, fmri2image}
  entity: ${oc.env:WANDB_ENTITY, null}
  mode: online
seed: 1337
device: cuda
run:
  name: baseline_train
  output_dir: outputs/${now:%Y-%m-%d}/${run.name}


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

# .dvc/cache/files/md5/b4/bb677d069190e8e3fafc8dc7bbaf09

```
[2025-09-29 22:39:01,379][__main__][INFO] - Config:
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
  enabled: false
  project: ${oc.env:WANDB_PROJECT, fmri2image}
  entity: ${oc.env:WANDB_ENTITY, null}
  mode: online
seed: 1337
device: cuda
run:
  name: baseline_train
  output_dir: outputs/${now:%Y-%m-%d}/${run.name}


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

# .dvc/cache/files/md5/c9/d45422aec828d538f162374e38999f.dir

```dir
[{"md5": "e2547361f1328fa9c8611fbb9217ffd3", "relpath": "2025-09-30/00-09-04/.hydra/config.yaml"}, {"md5": "fa21ed12ea7ec570645696debfcb7248", "relpath": "2025-09-30/00-09-04/.hydra/hydra.yaml"}, {"md5": "73bc718bee9b637ae12f8173a862007a", "relpath": "2025-09-30/00-09-04/.hydra/overrides.yaml"}, {"md5": "334da91462da50992134edf89d1c7eaf", "relpath": "2025-09-30/00-09-04/cli.log"}]
```

# .dvc/cache/files/md5/cb/2f2259e46435d7ee1a6000ddbfe736

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
  loss:
    type: contrastive
    temperature_init: 0.07
    symmetric: true
    weights:
      contrastive: 1.0
      cca: 0.05
  eval:
    topk:
    - 1
    - 5
  encoder: mlp
  gnn:
    use_identity_adj: true
    dropout: 0.1
  vit3d:
    time_steps: 8
    patch: 1
    depth: 2
    heads: 4
    mlp_ratio: 2.0
    dropout: 0.1
  cca:
    enabled: true
    proj_dim: 128
  pretrained:
    path: data/artifacts/encoder_pretrained.ckpt
eval:
  metrics:
  - ssim
  - psnr
  - clip_score
wandb:
  enabled: false
  project: ${oc.env:WANDB_PROJECT, fmri2image}
  entity: ${oc.env:WANDB_ENTITY, null}
  mode: online
seed: 1337
device: cuda
run:
  name: baseline_train
  output_dir: outputs/${now:%Y-%m-%d}/${run.name}

```

# .dvc/cache/files/md5/ce/ca2a359232b632d1d409d728ab4dec.dir

```dir
[{"md5": "d6b9288dce117311870f6a104fa9a647", "relpath": "2025-09-29/22-39-01/.hydra/config.yaml"}, {"md5": "7d18581bb9713d9ce2e02effda2a1ded", "relpath": "2025-09-29/22-39-01/.hydra/hydra.yaml"}, {"md5": "73bc718bee9b637ae12f8173a862007a", "relpath": "2025-09-29/22-39-01/.hydra/overrides.yaml"}, {"md5": "b4bb677d069190e8e3fafc8dc7bbaf09", "relpath": "2025-09-29/22-39-01/cli.log"}]
```

# .dvc/cache/files/md5/d6/b9288dce117311870f6a104fa9a647

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
  enabled: false
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

# .dvc/cache/files/md5/e2/547361f1328fa9c8611fbb9217ffd3

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
  loss:
    type: contrastive
    temperature_init: 0.07
    symmetric: true
    weights:
      contrastive: 1.0
      cca: 0.1
  eval:
    topk:
    - 1
    - 5
  encoder: mlp
  gnn:
    use_identity_adj: true
    dropout: 0.1
  vit3d:
    time_steps: 8
    patch: 1
    depth: 2
    heads: 4
    mlp_ratio: 2.0
    dropout: 0.1
  cca:
    enabled: true
    proj_dim: 128
  pretrained:
    path: data/artifacts/encoder_pretrained.ckpt
eval:
  metrics:
  - ssim
  - psnr
  - clip_score
wandb:
  enabled: false
  project: ${oc.env:WANDB_PROJECT, fmri2image}
  entity: ${oc.env:WANDB_ENTITY, null}
  mode: online
seed: 1337
device: cuda
run:
  name: baseline_train
  output_dir: outputs/${now:%Y-%m-%d}/${run.name}

```

# .dvc/cache/files/md5/e5/50133fa3ffe3a43dfdea89869a2362

```
[2025-09-29 22:28:31,435][__main__][INFO] - Config:
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
  enabled: false
  project: ${oc.env:WANDB_PROJECT, fmri2image}
  entity: ${oc.env:WANDB_ENTITY, null}
  mode: online
seed: 1337
device: cuda
run:
  name: baseline_train
  output_dir: outputs/${now:%Y-%m-%d}/${run.name}


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

# .dvc/cache/files/md5/fa/21ed12ea7ec570645696debfcb7248

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
    output_dir: /home/tonystark/Desktop/Bachelor/fmri2image/outputs/2025-09-30/00-09-04
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

# .dvc/cache/files/md5/fd/c9cab183abf9a24e255960effbc07f.dir

```dir
[{"md5": "d6b9288dce117311870f6a104fa9a647", "relpath": "2025-09-29/22-28-31/.hydra/config.yaml"}, {"md5": "727145e6dffb6fe0ee0aa77472970cc9", "relpath": "2025-09-29/22-28-31/.hydra/hydra.yaml"}, {"md5": "73bc718bee9b637ae12f8173a862007a", "relpath": "2025-09-29/22-28-31/.hydra/overrides.yaml"}, {"md5": "e550133fa3ffe3a43dfdea89869a2362", "relpath": "2025-09-29/22-28-31/cli.log"}]
```

# .dvc/cache/runs/6e/6e42d1946eaf8c129e702e2f02fc7926d5e03b7e81e6ed82eba51427d753e777/8b3d1895bd249bed8e46e67446db0e61b357d3187a0a76a25ef3af5cb18930c6

```
cmd: python -m fmri2image.cli ++run.name=baseline_train train=baseline
deps:
- path: configs/config.yaml
  hash: md5
  md5: c341062c96b325785ce81f82fb812de0
  size: 211
- path: configs/train/baseline.yaml
  hash: md5
  md5: 9005825d02ef13afb05055f967f6268f
  size: 434
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
  md5: 2bc217f4c92247f206831d0203181cdf
  size: 6184
outs:
- path: outputs
  hash: md5
  md5: fdc9cab183abf9a24e255960effbc07f.dir
  size: 6078
  nfiles: 4

```

# .dvc/cache/runs/51/51f0f7e239a54536e1c1f9a25b9ca2e21248b1f508a1260e50493a1ad0c0c659/3ec4341d6d14267bea9aecc7d10f2ee6ea1c4d37d84fabfd3f3d0730110535de

```
cmd: "python -m fmri2image.selfsupervised.pretrain_fmri --images_root data/raw/nsd/images
  --fmri_root data/raw/nsd/fmri --captions data/raw/nsd/captions.csv --roi_dir data/processed/nsd/roi
  --subject subj01 --dim 2048 --mask_ratio 0.5 --lr 1e-3 --epochs 2 --batch_size 16
  --num_workers 4 --out_ckpt data/artifacts/encoder_pretrained.ckpt\n"
deps:
- path: data/processed/nsd/roi/subj01_roi.npy
  hash: md5
  md5: 47c9fb596d852b19d0a97abc4504a6ae
  size: 983168
- path: src/fmri2image/data/nsd_reader.py
  hash: md5
  md5: 36d85d4ecab163fa022f4e5388d56dda
  size: 1110
- path: src/fmri2image/selfsupervised/pretrain_fmri.py
  hash: md5
  md5: 4f5b993d4bad0747b4dedd699e4ee9a5
  size: 3960
outs:
- path: data/artifacts/encoder_pretrained.ckpt
  hash: md5
  md5: 04a629f13e039b482e0b6686f20ab5a3
  size: 25180747

```

# .dvc/cache/runs/65/658fc4fe42d725c3c04d43b65752116642356a7f23ff0f31c6a447223436e284/d23d21a70bcb90761ce5539b67517bf445f9517fdb6b63a35ccfbed5ea489042

```
cmd: python -m fmri2image.cli ++run.name=baseline_train train=baseline
deps:
- path: configs/config.yaml
  hash: md5
  md5: c341062c96b325785ce81f82fb812de0
  size: 211
- path: configs/train/baseline.yaml
  hash: md5
  md5: 9005825d02ef13afb05055f967f6268f
  size: 434
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
  md5: 44dd1a3fdeef88fb8b9155483e7cc167
  size: 6240
outs:
- path: outputs
  hash: md5
  md5: ceca2a359232b632d1d409d728ab4dec.dir
  size: 6078
  nfiles: 4

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

# .dvc/cache/runs/c5/c5c317360028e226ff08efd780a7a330b30ba6c7afc9f681e0050c8210cf07ba/ee11a661dd6bc48df5c6a372557acf620ce453fd3c55e2a7497f15af50c29274

```
cmd: python -m fmri2image.cli ++run.name=baseline_train train=baseline
deps:
- path: configs/config.yaml
  hash: md5
  md5: c341062c96b325785ce81f82fb812de0
  size: 211
- path: configs/train/baseline.yaml
  hash: md5
  md5: 4424135d9170c50e26a916499bf1ba40
  size: 931
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
  md5: 4235ee8e77b289eed2ec57285a46fbab
  size: 9838
outs:
- path: outputs
  hash: md5
  md5: 69759c9ee2ed9b9fb9ba32a4d660985b.dir
  size: 6730
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

# .dvc/cache/runs/da/daa44b74a8e8b45256587fd37447a9d12e2b39f786ba29ab5f958f22d0ad2ca6/3d2a22e76e1b5fca41052b298c1589cf070a9c6ff41fd2ccd0ddf91fe21fafc0

```
cmd: python -m fmri2image.cli ++run.name=baseline_train train=baseline
deps:
- path: configs/config.yaml
  hash: md5
  md5: c341062c96b325785ce81f82fb812de0
  size: 211
- path: configs/train/baseline.yaml
  hash: md5
  md5: 34b0eb8bc73b4e86b785796d3dddc663
  size: 930
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
  md5: 4235ee8e77b289eed2ec57285a46fbab
  size: 9838
outs:
- path: outputs
  hash: md5
  md5: c9d45422aec828d538f162374e38999f.dir
  size: 6728
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

# .dvc/storage/files/md5/2c/fdc695fc4d02a0aef4e9c1185bf1ae

```
image_id,caption
0,a dog on grass
1,a red car
2,a mountain scene

```

# .dvc/storage/files/md5/04/a629f13e039b482e0b6686f20ab5a3

This is a binary file of the type: Binary

# .dvc/storage/files/md5/6d/90c392139922ff1e0a10240d3b1786

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
    output_dir: /home/tonystark/Desktop/Bachelor/fmri2image/outputs/2025-09-30/00-21-16
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

# .dvc/storage/files/md5/25/77ce444d57eb6c9b162b4a4c8b09f4

This is a binary file of the type: Binary

# .dvc/storage/files/md5/42/74adec5d9eaefdddf3e352e03fba64

```
NSD placeholder. Pentru date reale: configurați AWS CLI și sincronizați din bucketul NSD.
Struc.: raw/nsd/images/, raw/nsd/fmri/, raw/nsd/captions.csv

```

# .dvc/storage/files/md5/47/c9fb596d852b19d0a97abc4504a6ae

This is a binary file of the type: Binary

# .dvc/storage/files/md5/69/759c9ee2ed9b9fb9ba32a4d660985b.dir

```dir
[{"md5": "cb2f2259e46435d7ee1a6000ddbfe736", "relpath": "2025-09-30/00-21-16/.hydra/config.yaml"}, {"md5": "6d90c392139922ff1e0a10240d3b1786", "relpath": "2025-09-30/00-21-16/.hydra/hydra.yaml"}, {"md5": "73bc718bee9b637ae12f8173a862007a", "relpath": "2025-09-30/00-21-16/.hydra/overrides.yaml"}, {"md5": "77abf48e4a307692ce4ee6bf530dacce", "relpath": "2025-09-30/00-21-16/cli.log"}]
```

# .dvc/storage/files/md5/73/bc718bee9b637ae12f8173a862007a

```
- ++run.name=baseline_train
- train=baseline

```

# .dvc/storage/files/md5/77/abf48e4a307692ce4ee6bf530dacce

```
[2025-09-30 00:21:16,242][__main__][INFO] - Config:
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
    weights:
      contrastive: 1.0
      cca: 0.05
  eval:
    topk:
    - 1
    - 5
  encoder: mlp
  gnn:
    use_identity_adj: true
    dropout: 0.1
  vit3d:
    time_steps: 8
    patch: 1
    depth: 2
    heads: 4
    mlp_ratio: 2.0
    dropout: 0.1
  cca:
    enabled: true
    proj_dim: 128
  pretrained:
    path: data/artifacts/encoder_pretrained.ckpt
eval:
  metrics:
  - ssim
  - psnr
  - clip_score
wandb:
  enabled: false
  project: ${oc.env:WANDB_PROJECT, fmri2image}
  entity: ${oc.env:WANDB_ENTITY, null}
  mode: online
seed: 1337
device: cuda
run:
  name: baseline_train
  output_dir: outputs/${now:%Y-%m-%d}/${run.name}


```

# .dvc/storage/files/md5/79/a62e2ffa10ebce2562583f42158d68.dir

```dir
[{"md5": "4274adec5d9eaefdddf3e352e03fba64", "relpath": "README.txt"}, {"md5": "2cfdc695fc4d02a0aef4e9c1185bf1ae", "relpath": "captions.csv"}]
```

# .dvc/storage/files/md5/b1/14b4dbddd4b453946cb1a555809659.dir

```dir
[{"md5": "47c9fb596d852b19d0a97abc4504a6ae", "relpath": "subj01_roi.npy"}]
```

# .dvc/storage/files/md5/cb/2f2259e46435d7ee1a6000ddbfe736

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
  loss:
    type: contrastive
    temperature_init: 0.07
    symmetric: true
    weights:
      contrastive: 1.0
      cca: 0.05
  eval:
    topk:
    - 1
    - 5
  encoder: mlp
  gnn:
    use_identity_adj: true
    dropout: 0.1
  vit3d:
    time_steps: 8
    patch: 1
    depth: 2
    heads: 4
    mlp_ratio: 2.0
    dropout: 0.1
  cca:
    enabled: true
    proj_dim: 128
  pretrained:
    path: data/artifacts/encoder_pretrained.ckpt
eval:
  metrics:
  - ssim
  - psnr
  - clip_score
wandb:
  enabled: false
  project: ${oc.env:WANDB_PROJECT, fmri2image}
  entity: ${oc.env:WANDB_ENTITY, null}
  mode: online
seed: 1337
device: cuda
run:
  name: baseline_train
  output_dir: outputs/${now:%Y-%m-%d}/${run.name}

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
  weights:
    contrastive: 1.0
    cca: 0.05 # set 0.0 to disable contribution
eval:
  topk: [1, 5]

encoder: mlp # mlp | vit3d | gnn

gnn:
  use_identity_adj: true # dacă nu ai încă un A ROI disponibil
  dropout: 0.1

vit3d:
  time_steps: 8 # număr de “ferestre” temporale simulate (mock)
  patch: 1 # granularitate pe timp (mock)
  depth: 2
  heads: 4
  mlp_ratio: 2.0
  dropout: 0.1

cca:
  enabled: true
  proj_dim: 128

pretrained:
  path: data/artifacts/encoder_pretrained.ckpt # leave empty to skip

```

# configs/train/debug.yaml

```yaml
max_epochs: 1
batch_size: 1
num_workers: 0
precision: 32

```

# configs/train/pretrain.yaml

```yaml
pretrain:
  dim: ${train.model.fmri_input_dim}
  mask_ratio: 0.5
  lr: 1e-3
  epochs: 3
  batch_size: 32
  num_workers: 4
  out_ckpt: ${paths.artifacts_dir}/encoder_pretrained.ckpt

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
enabled: false # Switch to true to enable Weights & Biases logging
project: ${oc.env:WANDB_PROJECT, fmri2image}
entity: ${oc.env:WANDB_ENTITY, null}
mode: online

```

# data/artifacts/encoder_pretrained.ckpt

This is a binary file of the type: Binary

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
      md5: 4424135d9170c50e26a916499bf1ba40
      size: 931
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
      md5: 4235ee8e77b289eed2ec57285a46fbab
      size: 9838
    outs:
    - path: outputs
      hash: md5
      md5: 69759c9ee2ed9b9fb9ba32a4d660985b.dir
      size: 6730
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
  pretrain_encoder:
    cmd: "python -m fmri2image.selfsupervised.pretrain_fmri --images_root data/raw/nsd/images
      --fmri_root data/raw/nsd/fmri --captions data/raw/nsd/captions.csv --roi_dir
      data/processed/nsd/roi --subject subj01 --dim 2048 --mask_ratio 0.5 --lr 1e-3
      --epochs 2 --batch_size 16 --num_workers 4 --out_ckpt data/artifacts/encoder_pretrained.ckpt\n"
    deps:
    - path: data/processed/nsd/roi/subj01_roi.npy
      hash: md5
      md5: 47c9fb596d852b19d0a97abc4504a6ae
      size: 983168
    - path: src/fmri2image/data/nsd_reader.py
      hash: md5
      md5: 36d85d4ecab163fa022f4e5388d56dda
      size: 1110
    - path: src/fmri2image/selfsupervised/pretrain_fmri.py
      hash: md5
      md5: 4f5b993d4bad0747b4dedd699e4ee9a5
      size: 3960
    outs:
    - path: data/artifacts/encoder_pretrained.ckpt
      hash: md5
      md5: 04a629f13e039b482e0b6686f20ab5a3
      size: 25180747

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

  pretrain_encoder:
    cmd: >
      python -m fmri2image.selfsupervised.pretrain_fmri
      --images_root data/raw/nsd/images
      --fmri_root data/raw/nsd/fmri
      --captions data/raw/nsd/captions.csv
      --roi_dir data/processed/nsd/roi
      --subject subj01
      --dim 2048
      --mask_ratio 0.5
      --lr 1e-3
      --epochs 2
      --batch_size 16
      --num_workers 4
      --out_ckpt data/artifacts/encoder_pretrained.ckpt
    deps:
      - src/fmri2image/selfsupervised/pretrain_fmri.py
      - src/fmri2image/data/nsd_reader.py
      - data/processed/nsd/roi/subj01_roi.npy
    outs:
      - data/artifacts/encoder_pretrained.ckpt

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

# outputs/2025-09-30/00-21-16/.hydra/config.yaml

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
    weights:
      contrastive: 1.0
      cca: 0.05
  eval:
    topk:
    - 1
    - 5
  encoder: mlp
  gnn:
    use_identity_adj: true
    dropout: 0.1
  vit3d:
    time_steps: 8
    patch: 1
    depth: 2
    heads: 4
    mlp_ratio: 2.0
    dropout: 0.1
  cca:
    enabled: true
    proj_dim: 128
  pretrained:
    path: data/artifacts/encoder_pretrained.ckpt
eval:
  metrics:
  - ssim
  - psnr
  - clip_score
wandb:
  enabled: false
  project: ${oc.env:WANDB_PROJECT, fmri2image}
  entity: ${oc.env:WANDB_ENTITY, null}
  mode: online
seed: 1337
device: cuda
run:
  name: baseline_train
  output_dir: outputs/${now:%Y-%m-%d}/${run.name}

```

# outputs/2025-09-30/00-21-16/.hydra/hydra.yaml

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
    output_dir: /home/tonystark/Desktop/Bachelor/fmri2image/outputs/2025-09-30/00-21-16
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

# outputs/2025-09-30/00-21-16/.hydra/overrides.yaml

```yaml
- ++run.name=baseline_train
- train=baseline

```

# outputs/2025-09-30/00-21-16/cli.log

```log
[2025-09-30 00:21:16,242][__main__][INFO] - Config:
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
    weights:
      contrastive: 1.0
      cca: 0.05
  eval:
    topk:
    - 1
    - 5
  encoder: mlp
  gnn:
    use_identity_adj: true
    dropout: 0.1
  vit3d:
    time_steps: 8
    patch: 1
    depth: 2
    heads: 4
    mlp_ratio: 2.0
    dropout: 0.1
  cca:
    enabled: true
    proj_dim: 128
  pretrained:
    path: data/artifacts/encoder_pretrained.ckpt
eval:
  metrics:
  - ssim
  - psnr
  - clip_score
wandb:
  enabled: false
  project: ${oc.env:WANDB_PROJECT, fmri2image}
  entity: ${oc.env:WANDB_ENTITY, null}
  mode: online
seed: 1337
device: cuda
run:
  name: baseline_train
  output_dir: outputs/${now:%Y-%m-%d}/${run.name}


```

# outputs/2025-09-30/00-26-53/.hydra/config.yaml

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
    weights:
      contrastive: 1.0
      cca: 0.05
  eval:
    topk:
    - 1
    - 5
  encoder: mlp
  gnn:
    use_identity_adj: true
    dropout: 0.1
  vit3d:
    time_steps: 8
    patch: 1
    depth: 2
    heads: 4
    mlp_ratio: 2.0
    dropout: 0.1
  cca:
    enabled: true
    proj_dim: 128
  pretrained:
    path: data/artifacts/encoder_pretrained.ckpt
eval:
  metrics:
  - ssim
  - psnr
  - clip_score
wandb:
  enabled: false
  project: ${oc.env:WANDB_PROJECT, fmri2image}
  entity: ${oc.env:WANDB_ENTITY, null}
  mode: online
seed: 1337
device: cuda
run:
  name: baseline_debug
  output_dir: outputs/${now:%Y-%m-%d}/${run.name}

```

# outputs/2025-09-30/00-26-53/.hydra/hydra.yaml

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
    - train.encoder=mlp
  job:
    name: cli
    chdir: null
    override_dirname: train.encoder=mlp
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
    output_dir: /home/tonystark/Desktop/Bachelor/fmri2image/outputs/2025-09-30/00-26-53
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

# outputs/2025-09-30/00-26-53/.hydra/overrides.yaml

```yaml
- train.encoder=mlp

```

# outputs/2025-09-30/00-26-53/cli.log

```log
[2025-09-30 00:26:53,065][__main__][INFO] - Config:
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
    weights:
      contrastive: 1.0
      cca: 0.05
  eval:
    topk:
    - 1
    - 5
  encoder: mlp
  gnn:
    use_identity_adj: true
    dropout: 0.1
  vit3d:
    time_steps: 8
    patch: 1
    depth: 2
    heads: 4
    mlp_ratio: 2.0
    dropout: 0.1
  cca:
    enabled: true
    proj_dim: 128
  pretrained:
    path: data/artifacts/encoder_pretrained.ckpt
eval:
  metrics:
  - ssim
  - psnr
  - clip_score
wandb:
  enabled: false
  project: ${oc.env:WANDB_PROJECT, fmri2image}
  entity: ${oc.env:WANDB_ENTITY, null}
  mode: online
seed: 1337
device: cuda
run:
  name: baseline_debug
  output_dir: outputs/${now:%Y-%m-%d}/${run.name}


```

# outputs/2025-09-30/00-26-59/.hydra/config.yaml

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
    weights:
      contrastive: 1.0
      cca: 0.05
  eval:
    topk:
    - 1
    - 5
  encoder: vit3d
  gnn:
    use_identity_adj: true
    dropout: 0.1
  vit3d:
    time_steps: 8
    patch: 1
    depth: 2
    heads: 4
    mlp_ratio: 2.0
    dropout: 0.1
  cca:
    enabled: true
    proj_dim: 128
  pretrained:
    path: data/artifacts/encoder_pretrained.ckpt
eval:
  metrics:
  - ssim
  - psnr
  - clip_score
wandb:
  enabled: false
  project: ${oc.env:WANDB_PROJECT, fmri2image}
  entity: ${oc.env:WANDB_ENTITY, null}
  mode: online
seed: 1337
device: cuda
run:
  name: baseline_debug
  output_dir: outputs/${now:%Y-%m-%d}/${run.name}

```

# outputs/2025-09-30/00-26-59/.hydra/hydra.yaml

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
    - train.encoder=vit3d
  job:
    name: cli
    chdir: null
    override_dirname: train.encoder=vit3d
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
    output_dir: /home/tonystark/Desktop/Bachelor/fmri2image/outputs/2025-09-30/00-26-59
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

# outputs/2025-09-30/00-26-59/.hydra/overrides.yaml

```yaml
- train.encoder=vit3d

```

# outputs/2025-09-30/00-26-59/cli.log

```log
[2025-09-30 00:27:00,004][__main__][INFO] - Config:
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
    weights:
      contrastive: 1.0
      cca: 0.05
  eval:
    topk:
    - 1
    - 5
  encoder: vit3d
  gnn:
    use_identity_adj: true
    dropout: 0.1
  vit3d:
    time_steps: 8
    patch: 1
    depth: 2
    heads: 4
    mlp_ratio: 2.0
    dropout: 0.1
  cca:
    enabled: true
    proj_dim: 128
  pretrained:
    path: data/artifacts/encoder_pretrained.ckpt
eval:
  metrics:
  - ssim
  - psnr
  - clip_score
wandb:
  enabled: false
  project: ${oc.env:WANDB_PROJECT, fmri2image}
  entity: ${oc.env:WANDB_ENTITY, null}
  mode: online
seed: 1337
device: cuda
run:
  name: baseline_debug
  output_dir: outputs/${now:%Y-%m-%d}/${run.name}


```

# outputs/2025-09-30/00-27-04/.hydra/config.yaml

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
    weights:
      contrastive: 1.0
      cca: 0.05
  eval:
    topk:
    - 1
    - 5
  encoder: gnn
  gnn:
    use_identity_adj: true
    dropout: 0.1
  vit3d:
    time_steps: 8
    patch: 1
    depth: 2
    heads: 4
    mlp_ratio: 2.0
    dropout: 0.1
  cca:
    enabled: true
    proj_dim: 128
  pretrained:
    path: data/artifacts/encoder_pretrained.ckpt
eval:
  metrics:
  - ssim
  - psnr
  - clip_score
wandb:
  enabled: false
  project: ${oc.env:WANDB_PROJECT, fmri2image}
  entity: ${oc.env:WANDB_ENTITY, null}
  mode: online
seed: 1337
device: cuda
run:
  name: baseline_debug
  output_dir: outputs/${now:%Y-%m-%d}/${run.name}

```

# outputs/2025-09-30/00-27-04/.hydra/hydra.yaml

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
    - train.encoder=gnn
  job:
    name: cli
    chdir: null
    override_dirname: train.encoder=gnn
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
    output_dir: /home/tonystark/Desktop/Bachelor/fmri2image/outputs/2025-09-30/00-27-04
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

# outputs/2025-09-30/00-27-04/.hydra/overrides.yaml

```yaml
- train.encoder=gnn

```

# outputs/2025-09-30/00-27-04/cli.log

```log
[2025-09-30 00:27:04,314][__main__][INFO] - Config:
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
    weights:
      contrastive: 1.0
      cca: 0.05
  eval:
    topk:
    - 1
    - 5
  encoder: gnn
  gnn:
    use_identity_adj: true
    dropout: 0.1
  vit3d:
    time_steps: 8
    patch: 1
    depth: 2
    heads: 4
    mlp_ratio: 2.0
    dropout: 0.1
  cca:
    enabled: true
    proj_dim: 128
  pretrained:
    path: data/artifacts/encoder_pretrained.ckpt
eval:
  metrics:
  - ssim
  - psnr
  - clip_score
wandb:
  enabled: false
  project: ${oc.env:WANDB_PROJECT, fmri2image}
  entity: ${oc.env:WANDB_ENTITY, null}
  mode: online
seed: 1337
device: cuda
run:
  name: baseline_debug
  output_dir: outputs/${now:%Y-%m-%d}/${run.name}


```

# outputs/2025-09-30/00-29-16/.hydra/config.yaml

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
  batch_size: 4
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
    weights:
      contrastive: 1.0
      cca: 0.05
  eval:
    topk:
    - 1
    - 5
  encoder: vit3d
  gnn:
    use_identity_adj: true
    dropout: 0.1
  vit3d:
    time_steps: 8
    patch: 1
    depth: 2
    heads: 4
    mlp_ratio: 2.0
    dropout: 0.1
  cca:
    enabled: true
    proj_dim: 128
  pretrained:
    path: data/artifacts/encoder_pretrained.ckpt
eval:
  metrics:
  - ssim
  - psnr
  - clip_score
wandb:
  enabled: false
  project: ${oc.env:WANDB_PROJECT, fmri2image}
  entity: ${oc.env:WANDB_ENTITY, null}
  mode: online
seed: 1337
device: cuda
run:
  name: baseline_debug
  output_dir: outputs/${now:%Y-%m-%d}/${run.name}

```

# outputs/2025-09-30/00-29-16/.hydra/hydra.yaml

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
    - train.encoder=vit3d
    - train.batch_size=4
  job:
    name: cli
    chdir: null
    override_dirname: train.batch_size=4,train.encoder=vit3d
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
    output_dir: /home/tonystark/Desktop/Bachelor/fmri2image/outputs/2025-09-30/00-29-16
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

# outputs/2025-09-30/00-29-16/.hydra/overrides.yaml

```yaml
- train.encoder=vit3d
- train.batch_size=4

```

# outputs/2025-09-30/00-29-16/cli.log

```log
[2025-09-30 00:29:17,024][__main__][INFO] - Config:
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
  batch_size: 4
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
    weights:
      contrastive: 1.0
      cca: 0.05
  eval:
    topk:
    - 1
    - 5
  encoder: vit3d
  gnn:
    use_identity_adj: true
    dropout: 0.1
  vit3d:
    time_steps: 8
    patch: 1
    depth: 2
    heads: 4
    mlp_ratio: 2.0
    dropout: 0.1
  cca:
    enabled: true
    proj_dim: 128
  pretrained:
    path: data/artifacts/encoder_pretrained.ckpt
eval:
  metrics:
  - ssim
  - psnr
  - clip_score
wandb:
  enabled: false
  project: ${oc.env:WANDB_PROJECT, fmri2image}
  entity: ${oc.env:WANDB_ENTITY, null}
  mode: online
seed: 1337
device: cuda
run:
  name: baseline_debug
  output_dir: outputs/${now:%Y-%m-%d}/${run.name}


```

# outputs/2025-09-30/00-29-33/.hydra/config.yaml

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
    weights:
      contrastive: 1.0
      cca: 0.1
  eval:
    topk:
    - 1
    - 5
  encoder: mlp
  gnn:
    use_identity_adj: true
    dropout: 0.1
  vit3d:
    time_steps: 8
    patch: 1
    depth: 2
    heads: 4
    mlp_ratio: 2.0
    dropout: 0.1
  cca:
    enabled: true
    proj_dim: 128
  pretrained:
    path: data/artifacts/encoder_pretrained.ckpt
eval:
  metrics:
  - ssim
  - psnr
  - clip_score
wandb:
  enabled: false
  project: ${oc.env:WANDB_PROJECT, fmri2image}
  entity: ${oc.env:WANDB_ENTITY, null}
  mode: online
seed: 1337
device: cuda
run:
  name: baseline_debug
  output_dir: outputs/${now:%Y-%m-%d}/${run.name}

```

# outputs/2025-09-30/00-29-33/.hydra/hydra.yaml

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
    - train.loss.weights.cca=0.1
  job:
    name: cli
    chdir: null
    override_dirname: train.loss.weights.cca=0.1
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
    output_dir: /home/tonystark/Desktop/Bachelor/fmri2image/outputs/2025-09-30/00-29-33
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

# outputs/2025-09-30/00-29-33/.hydra/overrides.yaml

```yaml
- train.loss.weights.cca=0.1

```

# outputs/2025-09-30/00-29-33/cli.log

```log
[2025-09-30 00:29:33,540][__main__][INFO] - Config:
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
    weights:
      contrastive: 1.0
      cca: 0.1
  eval:
    topk:
    - 1
    - 5
  encoder: mlp
  gnn:
    use_identity_adj: true
    dropout: 0.1
  vit3d:
    time_steps: 8
    patch: 1
    depth: 2
    heads: 4
    mlp_ratio: 2.0
    dropout: 0.1
  cca:
    enabled: true
    proj_dim: 128
  pretrained:
    path: data/artifacts/encoder_pretrained.ckpt
eval:
  metrics:
  - ssim
  - psnr
  - clip_score
wandb:
  enabled: false
  project: ${oc.env:WANDB_PROJECT, fmri2image}
  entity: ${oc.env:WANDB_ENTITY, null}
  mode: online
seed: 1337
device: cuda
run:
  name: baseline_debug
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
from typing import Sequence, Tuple
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np


class FMRITextDataset(Dataset):
    """
    Minimal dataset that returns:
      - X[idx]: fMRI feature vector as float32 tensor
      - (idx, text): sample index (for CLIP alignment) and raw caption
    """
    def __init__(self, X: np.ndarray, texts: Sequence[str]):
        assert len(X) == len(texts), "X and texts must have the same length"
        self.X = X
        self.texts = list(texts)

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Tuple[torch.Tensor, str]]:
        # Ensure float32 for model stability
        x = torch.tensor(self.X[idx], dtype=torch.float32)
        i = torch.tensor(idx, dtype=torch.long)
        t = self.texts[idx]
        return x, (i, t)


def make_loaders(
    X: np.ndarray,
    texts: Sequence[str],
    batch_size: int = 2,
    num_workers: int = 0,
    *,
    drop_last: bool = True,         # <-- important for CCA stability
    pin_memory: bool = True,
):
    """
    Returns a single train DataLoader. We drop the last batch to avoid B=1,
    which can cause numerical issues in CCA/statistics.
    """
    ds = FMRITextDataset(X, texts)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=drop_last,                 # <-- key change
        pin_memory=pin_memory,
        persistent_workers=bool(num_workers) # keeps workers alive if >0
    )
    return loader

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

# src/fmri2image/generative/adapters.py

```py
import torch
import torch.nn as nn

class LoRALinear(nn.Module):
    """
    Classic LoRA module: y = xW + alpha/ r * xBA
    W is frozen (assumed in parent), only A,B are trainable.
    """
    def __init__(self, base_linear: nn.Linear, rank: int = 8, alpha: float = 1.0):
        super().__init__()
        assert isinstance(base_linear, nn.Linear)
        self.base = base_linear
        for p in self.base.parameters():
            p.requires_grad = False
        self.rank = rank
        self.alpha = alpha
        in_f, out_f = base_linear.in_features, base_linear.out_features
        self.A = nn.Linear(in_f, rank, bias=False)
        self.B = nn.Linear(rank, out_f, bias=False)
        nn.init.kaiming_uniform_(self.A.weight, a=5**0.5)
        nn.init.zeros_(self.B.weight)

    def forward(self, x):
        return self.base(x) + (self.alpha / self.rank) * self.B(self.A(x))

```

# src/fmri2image/losses/cca.py

```py
import torch
import torch.nn as nn

class DeepCCALoss(nn.Module):
    """
    Safe Deep CCA-style auxiliary loss:
    - proiectează z și t în același spațiu
    - centrează + standardizează cu unbiased=False
    - dacă batch < 2, sare peste (loss=0)
    - maximizează corelația medie (loss = -mean|corr|)
    """
    def __init__(self, in_dim: int, out_dim: int, eps: float = 1e-6):
        super().__init__()
        self.proj_z = nn.Linear(in_dim, out_dim)
        self.proj_t = nn.Linear(in_dim, out_dim)
        self.eps = float(eps)

    def _standardize(self, x: torch.Tensor) -> torch.Tensor:
        # center + scale; unbiased=False ca să nu ceară d.o.f. >= 1
        x = x - x.mean(dim=0, keepdim=True)
        std = x.std(dim=0, keepdim=True, unbiased=False)
        std = torch.clamp(std, min=self.eps)
        return x / std

    def forward(self, z: torch.Tensor, t: torch.Tensor):
        B = z.size(0)
        device = z.device
        if B < 2:
            # prea puține eșantioane pentru o estimare rezonabilă
            zero = torch.zeros((), device=device, dtype=z.dtype)
            return {"loss": zero, "corr_sum": zero}

        z_p = self.proj_z(z)
        t_p = self.proj_t(t)

        z_p = self._standardize(z_p)
        t_p = self._standardize(t_p)

        # corelație pe feature-uri, apoi media pe feature-uri
        # echivalent cu cosine pe coloane (după standardizare)
        corr_per_dim = (z_p * t_p).mean(dim=0)           # [D]
        corr_mean = corr_per_dim.abs().mean()            # scalar

        loss = -corr_mean
        return {"loss": loss, "corr_sum": corr_mean}

```

# src/fmri2image/models/encoders/gnn_encoder.py

```py
import torch
import torch.nn as nn

class GraphMLPEncoderLite(nn.Module):
    """
    Minimal graph-like encoder using adjacency via (I + D^{-1/2} A D^{-1/2})
    applied as a fixed linear propagation step, followed by MLP.
    Avoids external deps (PyG). Works with ROI vectors shaped [B, D].
    """
    def __init__(self, in_dim: int, out_dim: int, hidden: list[int], dropout: float = 0.1):
        super().__init__()
        dims = [in_dim] + list(hidden) + [out_dim]
        layers = []
        for i in range(len(dims) - 1):
            layers += [nn.Linear(dims[i], dims[i + 1])]
            if i < len(dims) - 2:
                layers += [nn.ReLU(), nn.Dropout(dropout)]
        self.mlp = nn.Sequential(*layers)
        self.dropout = nn.Dropout(dropout)

        # adjacency placeholder (registered later via set_adj)
        self.register_buffer("prop_matrix", None, persistent=False)

    def set_adj(self, adj: torch.Tensor):
        """
        adj: [D, D] unweighted/weighted adjacency (0/1 or weights).
        Builds normalized propagation matrix P = I + D^{-1/2} A D^{-1/2}.
        """
        A = adj
        I = torch.eye(A.size(0), device=A.device, dtype=A.dtype)
        deg = A.sum(dim=1)
        deg_inv_sqrt = torch.pow(deg + 1e-6, -0.5)
        D_inv_sqrt = torch.diag(deg_inv_sqrt)
        P = I + D_inv_sqrt @ A @ D_inv_sqrt
        self.prop_matrix = P

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, D]; apply one graph propagation x' = x P, then MLP.
        If no adj given, just MLP.
        """
        if self.prop_matrix is not None:
            x = x @ self.prop_matrix
        x = self.dropout(x)
        return self.mlp(x)

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

# src/fmri2image/models/encoders/vit3d_encoder.py

```py
import torch
import torch.nn as nn

class ViT3DEncoderLite(nn.Module):
    """
    A very small '3D ViT' surrogate for ablations, operating on
    fMRI ROI vectors by reshaping the feature dim into T x token_dim.
    In real fMRI 4D you'd feed [B, T, V] or [B, T, H, W, D] tokens.
    """
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        time_steps: int = 8,
        depth: int = 2,
        heads: int = 4,
        mlp_ratio: float = 2.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert in_dim % time_steps == 0, "in_dim must be divisible by time_steps for the lite reshaping"
        self.time_steps = int(time_steps)
        token_dim = in_dim // self.time_steps
        self.token_proj = nn.Linear(token_dim, token_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=token_dim,
            nhead=heads,
            dim_feedforward=int(token_dim * mlp_ratio),
            dropout=dropout,
            batch_first=True,
        )
        self.backbone = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.head = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, in_dim] -> reshape to [B, T, token_dim], run transformer over T,
        pool (mean) and map to out_dim.
        """
        B, D = x.shape
        T = self.time_steps
        token_dim = D // T
        xt = x.view(B, T, token_dim)
        xt = self.token_proj(xt)
        xt = self.backbone(xt)   # [B, T, token_dim]
        xt = xt.mean(dim=1)      # temporal mean pool
        z = self.head(xt)        # [B, out_dim]
        return z

```

# src/fmri2image/pipelines/baseline_train.py

```py
from omegaconf import DictConfig
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl

from ..models.encoders.mlp_encoder import FMRIEncoderMLP
from ..models.encoders.vit3d_encoder import ViT3DEncoderLite
from ..models.encoders.gnn_encoder import GraphMLPEncoderLite

from ..losses.cca import DeepCCALoss

from ..data.nsd_reader import NSDReader
from ..data.datamodule import make_loaders

from .metrics import topk_retrieval


def build_encoder(cfg: DictConfig, out_dim: int) -> nn.Module:
    """
    Select and construct the encoder according to cfg.train.encoder.
    Supported: 'mlp' (default), 'vit3d', 'gnn'.
    """
    enc_type = str(getattr(cfg.train, "encoder", "mlp")).lower()
    m = cfg.train.model

    if enc_type == "mlp":
        return FMRIEncoderMLP(m.fmri_input_dim, out_dim, m.hidden)

    elif enc_type == "vit3d":
        v = cfg.train.vit3d
        # NOTE: ViT3D-lite expects fmri_input_dim % time_steps == 0
        return ViT3DEncoderLite(
            in_dim=m.fmri_input_dim,
            out_dim=out_dim,
            time_steps=int(getattr(v, "time_steps", 8)),
            depth=int(getattr(v, "depth", 2)),
            heads=int(getattr(v, "heads", 4)),
            mlp_ratio=float(getattr(v, "mlp_ratio", 2.0)),
            dropout=float(getattr(v, "dropout", 0.1)),
        )

    elif enc_type == "gnn":
        g = cfg.train.gnn
        enc = GraphMLPEncoderLite(
            in_dim=m.fmri_input_dim,
            out_dim=out_dim,
            hidden=m.hidden,
            dropout=float(getattr(g, "dropout", 0.1)),
        )
        # Mock adjacency: identity (no edges) unless you switch to another scheme
        use_identity = bool(getattr(g, "use_identity_adj", True))
        if use_identity:
            A = torch.zeros(m.fmri_input_dim, m.fmri_input_dim)
        else:
            # simple example: self-loops (you can customize later)
            A = torch.eye(m.fmri_input_dim, m.fmri_input_dim)
        enc.set_adj(A)
        return enc

    else:
        raise ValueError(f"Unknown encoder type: {enc_type}")


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

        # Project fMRI -> CLIP text embedding dimension
        out_dim = int(clip_text_feats.shape[1])
        self.encoder = build_encoder(cfg, out_dim)

        # Contrastive loss
        self.criterion = ClipStyleContrastiveLoss(
            temperature_init=float(cfg.train.loss.temperature_init),
            symmetric=bool(cfg.train.loss.get("symmetric", True)),
        )

        # Optional Deep CCA
        cca_cfg = getattr(getattr(cfg, "train", {}), "cca", {})
        self.use_cca = bool(getattr(cca_cfg, "enabled", False))
        if self.use_cca:
            cca_out_dim = int(getattr(cca_cfg, "proj_dim", 128))
            self.cca = DeepCCALoss(in_dim=out_dim, out_dim=cca_out_dim)
        else:
            self.cca = None

        # loss weights
        w = getattr(cfg.train.loss, "weights", {})
        self.w_contrast = float(w.get("contrastive", 1.0))
        self.w_cca = float(w.get("cca", 0.0))

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

        # --- Contrastive ---
        out = self.criterion(z, t)
        loss_total = self.w_contrast * out["loss"]

        # --- Optional Deep CCA (auxiliary) ---
        if self.use_cca and self.w_cca > 0:
            cca_out = self.cca(z.detach(), t.detach())  # keep it auxiliary; no grad through encoder
            loss_total = loss_total + self.w_cca * cca_out["loss"]
        else:
            cca_out = None

        # --- Logging with explicit batch_size ---
        bs = x.size(0)
        self.log("train/loss", loss_total, prog_bar=True, on_step=True, on_epoch=True, batch_size=bs)
        self.log("train/loss_contrast", out["loss"], prog_bar=False, on_step=True, on_epoch=True, batch_size=bs)
        self.log("train/temp", out["temp"], prog_bar=False, on_step=True, on_epoch=True, batch_size=bs)
        if cca_out is not None:
            self.log("train/cca_corr", cca_out["corr_sum"], prog_bar=False, on_step=True, on_epoch=True, batch_size=bs)

        # --- Retrieval metrics within-batch (ranking unaffected by temperature) ---
        with torch.no_grad():
            # remove temperature for ranking
            sim_zt = out["logits_zt"] / torch.exp(self.criterion.logit_scale)
            m_zt = topk_retrieval(sim_zt, self.topk)
            for k, v in m_zt.items():
                self.log(f"train/retrieval_zt_{k}", v, prog_bar=True, on_step=False, on_epoch=True, batch_size=bs)

            if out["logits_tz"] is not None:
                sim_tz = out["logits_tz"] / torch.exp(self.criterion.logit_scale)
                m_tz = topk_retrieval(sim_tz, self.topk)
                for k, v in m_tz.items():
                    self.log(f"train/retrieval_tz_{k}", v, prog_bar=False, on_step=False, on_epoch=True, batch_size=bs)

        return loss_total

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

    # Optionally load self-supervised pretrained encoder weights
    pre_ckpt = getattr(getattr(cfg.train, "pretrained", {}), "path", None)
    if pre_ckpt and os.path.exists(pre_ckpt):
        ckpt = torch.load(pre_ckpt, map_location="cpu")
        enc_state = ckpt.get("state_dict", ckpt)
        missing, unexpected = model.encoder.load_state_dict(enc_state, strict=False)
        print(f"[pretrain] loaded encoder weights from {pre_ckpt} "
              f"(missing={len(missing)}, unexpected={len(unexpected)})")

    # Optional: logger (W&B disabled by default in your config)
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

# src/fmri2image/selfsupervised/pretrain_fmri.py

```py
from __future__ import annotations
import argparse
import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import TensorDataset, DataLoader

from omegaconf import OmegaConf
from fmri2image.data.nsd_reader import NSDReader

# Simple masked autoencoder over ROI vector (no time)
class MaskedVectorAE(nn.Module):
    def __init__(self, dim: int, hidden: list[int] = [2048, 1024]):
        super().__init__()
        enc_dims = [dim] + hidden
        dec_dims = hidden[::-1] + [dim]
        enc = []
        for i in range(len(enc_dims) - 1):
            enc += [nn.Linear(enc_dims[i], enc_dims[i+1]), nn.ReLU()]
        dec = []
        for i in range(len(dec_dims) - 1):
            dec += [nn.Linear(dec_dims[i], dec_dims[i+1])]
            if i < len(dec_dims) - 2:
                dec += [nn.ReLU()]
        self.encoder = nn.Sequential(*enc)
        self.decoder = nn.Sequential(*dec)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat

class LitMaskedAE(pl.LightningModule):
    def __init__(self, dim: int, mask_ratio: float = 0.5, lr: float = 1e-3, hidden=[2048,1024]):
        super().__init__()
        self.save_hyperparameters()
        self.model = MaskedVectorAE(dim, hidden)
        self.mask_ratio = mask_ratio
        self.criterion = nn.MSELoss()
        self.lr = lr

    def training_step(self, batch, _):
        (x,) = batch
        B, D = x.shape
        # random mask per sample
        k = int(D * self.mask_ratio)
        idx = torch.rand(B, D, device=x.device).argsort(dim=1)
        mask = torch.ones_like(x)
        mask.scatter_(1, idx[:, :k], 0.0)  # masked positions -> 0
        x_masked = x * mask

        x_hat = self.model(x_masked)
        loss = self.criterion(x_hat * (1 - mask), x * (1 - mask))  # reconstruct only masked
        self.log("pretrain/mse_masked", loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=B)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--images_root", type=str, default="data/raw/nsd/images")
    p.add_argument("--fmri_root", type=str, default="data/raw/nsd/fmri")
    p.add_argument("--captions", type=str, default="data/raw/nsd/captions.csv")
    p.add_argument("--roi_dir", type=str, default="data/processed/nsd/roi")
    p.add_argument("--subject", type=str, default="subj01")
    p.add_argument("--dim", type=int, default=2048)
    p.add_argument("--mask_ratio", type=float, default=0.5)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--out_ckpt", type=str, default="data/artifacts/encoder_pretrained.ckpt")
    args = p.parse_args()

    # Load ROI vectors (mock)
    reader = NSDReader(args.images_root, args.fmri_root, args.captions,
                       roi_dir=args.roi_dir, subject=args.subject)
    X, _ = reader.load(n=1024, fmri_dim=args.dim)  # more samples for pretrain
    X = torch.tensor(X, dtype=torch.float32)

    ds = TensorDataset(X)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    lit = LitMaskedAE(dim=args.dim, mask_ratio=args.mask_ratio, lr=args.lr)
    trainer = pl.Trainer(max_epochs=args.epochs, enable_checkpointing=False, logger=False)
    trainer.fit(lit, dl)

    # Save state dict so we can load encoder weights later
    ckpt = {"state_dict": lit.model.encoder.state_dict(), "dim": args.dim}
    out_path = args.out_ckpt
    import os
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save(ckpt, out_path)
    print(f"[ok] saved pretrained encoder -> {out_path}")

if __name__ == "__main__":
    main()

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

# tools/run_ablations.py

```py
#!/usr/bin/env python3
"""
Run ablations for encoders: mlp, vit3d, gnn
- compune config-ul Hydra (fără a schimba cwd)
- rulează training cu CSVLogger
- citește metri ce din csv și construiește summary.csv

Rezultate:
- outputs/ablations/<encoder>/metrics.csv         (Lightning CSV)
- outputs/ablations/summary.csv                   (rezumat)
"""
from __future__ import annotations

import os
import csv
import time
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import torch
import pytorch_lightning as pl

# Hydra + OmegaConf
from hydra import initialize_config_dir, compose
from omegaconf import OmegaConf, DictConfig

# Importăm direct componentele din pipeline-ul tău
from fmri2image.pipelines.baseline_train import (
    LitModule,
    build_encoder,   # not used directly, kept for clarity
)
from fmri2image.data.nsd_reader import NSDReader
from fmri2image.data.datamodule import make_loaders


def run_once(cfg: DictConfig, encoder_name: str, out_root: Path) -> Dict[str, float]:
    """
    Rulează un training pentru encoderul 'encoder_name' și returnează metricele finale.
    """
    # Override encoder în config (fără a pierde restul)
    cfg = OmegaConf.merge(cfg, OmegaConf.create({"train": {"encoder": encoder_name}}))

    # ==== Data ====
    reader = NSDReader(
        cfg.data.paths.images_root,
        cfg.data.paths.fmri_root,
        cfg.data.paths.captions,
        roi_dir=cfg.data.roi.out_dir,
        subject=cfg.data.subjects[0] if "subjects" in cfg.data and cfg.data.subjects else "subj01",
    )
    X, texts = reader.load(n=64, fmri_dim=cfg.train.model.fmri_input_dim)

    clip_feats_path = Path("data/processed/nsd/clip_text.npy")
    if not clip_feats_path.exists():
        raise FileNotFoundError(f"Missing CLIP text features at: {clip_feats_path}")
    clip_feats = np.load(clip_feats_path)

    n = min(len(X), len(clip_feats))
    X, texts, clip_feats = X[:n], texts[:n], clip_feats[:n]

    dl = make_loaders(X, texts, cfg.train.batch_size, cfg.train.num_workers)

    # ==== Model ====
    model = LitModule(cfg, clip_feats)

    # ==== Logger CSV (un folder per encoder) ====
    # outputs/ablations/<encoder>/
    save_dir = out_root / encoder_name
    save_dir.mkdir(parents=True, exist_ok=True)
    from pytorch_lightning.loggers import CSVLogger
    csv_logger = CSVLogger(save_dir=str(save_dir), name="", version=".")

    # ==== Trainer ====
    trainer = pl.Trainer(
        max_epochs=cfg.train.max_epochs,
        precision=cfg.train.precision,
        default_root_dir=str(save_dir),
        enable_checkpointing=False,
        logger=csv_logger,
    )
    trainer.fit(model, dl)

    # ==== Citește ultimele valori din CSV ====
    metrics_file = save_dir / "metrics.csv"
    if not metrics_file.exists():
        raise FileNotFoundError(f"Expected metrics.csv at {metrics_file}")

    last_row: Dict[str, Any] = {}
    with metrics_file.open("r", newline="") as f:
        reader_csv = csv.DictReader(f)
        rows = list(reader_csv)
        if not rows:
            raise RuntimeError(f"No rows in {metrics_file}")
        last_row = rows[-1]

    # Extrage chei utile; fallback la None dacă lipsesc
    def get_float(key: str) -> float | None:
        val = last_row.get(key, None)
        try:
            return float(val) if val is not None and val != "" else None
        except Exception:
            return None

    summary = {
        "encoder": encoder_name,
        "train/loss_epoch": get_float("train/loss_epoch"),
        "train/retrieval_zt_top1": get_float("train/retrieval_zt_top1_epoch"),
        "train/retrieval_zt_top5": get_float("train/retrieval_zt_top5_epoch"),
        # temperatură medie pe epocă (opțional)
        "train/temp_epoch": get_float("train/temp_epoch"),
    }
    return summary


def main():
    # Respectă WANDB_DISABLED by default
    os.environ.setdefault("WANDB_DISABLED", "true")

    # Compunem config-ul principal folosind Hydra fără a schimba CWD
    # `config_name="config"` -> configs/config.yaml
    this_file = Path(__file__).resolve()
    repo_root = this_file.parents[1]  # .../fmri2image
    configs_dir = repo_root / "configs"

    if not configs_dir.exists():
        raise FileNotFoundError(f"Configs directory not found at {configs_dir}")

    # Hidra: nu schimba directorul de lucru
    with initialize_config_dir(config_dir=str(configs_dir), version_base=None):
        cfg = compose(config_name="config")
        cfg = OmegaConf.to_container(cfg, resolve=True)
        cfg = OmegaConf.create(cfg)

    out_root = repo_root / "outputs" / "ablations"
    out_root.mkdir(parents=True, exist_ok=True)

    encoders = ["mlp", "vit3d", "gnn"]
    results: List[Dict[str, Any]] = []

    for enc in encoders:
        print(f"\n[ABLATION] Running encoder={enc} ...")
        res = run_once(cfg, enc, out_root)
        print(f"[ABLATION] Done encoder={enc}: "
              f"loss={res.get('train/loss_epoch')}, "
              f"top1={res.get('train/retrieval_zt_top1')}, "
              f"top5={res.get('train/retrieval_zt_top5')}")
        results.append(res)

    # Scrie summary.csv
    summary_path = out_root / "summary.csv"
    fieldnames = ["encoder", "train/loss_epoch", "train/retrieval_zt_top1",
                  "train/retrieval_zt_top5", "train/temp_epoch"]
    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    print(f"\n[ABLATION] Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()

```

