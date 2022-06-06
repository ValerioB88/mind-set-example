import numpy as np
import torch
import neptune.new as neptune
from sty import ef, rs, fg, bg

class Config:
    def __init__(self, **kwargs):
        self.use_cuda = torch.cuda.is_available()
        self.verbose = True
        [self.__setattr__(k, v) for k, v in kwargs.items()]
        self.finalize_init(**kwargs)

    def __setattr__(self, *args, **kwargs):
        if hasattr(self, 'weblogger'):
            if isinstance(self.weblogger, neptune.Run):
                self.weblogger[f"parameters/{args[0]}"] = str(args[1])
        super().__setattr__(*args, **kwargs)

    def finalize_init(self, **PARAMS):
        if 'list_tags' in PARAMS:
            list_tags = PARAMS['list_tags']
        else:
            list_tags = []
        print(fg.magenta)
        print('**LIST_TAGS**:')
        print(list_tags)
        if self.verbose:
            print('***PARAMS***')
            if not self.use_cuda:
                list_tags.append('LOCALTEST')

            for i in sorted(PARAMS.keys()):
                print(f'\t{i} : ' + ef.inverse + f'{PARAMS[i]}' + rs.inverse)

        if self.weblogger:
            try:
                neptune_run = neptune.init(f'valeriobiscione/{self.project_name}')
                neptune_run["sys/tags"].add(list_tags)
                neptune_run["parameters"] = PARAMS
                self.weblogger = neptune_run
            except:
                print("Initializing neptune didn't work, maybe you don't have neptune installed or you haven't set up the API token (https://docs.neptune.ai/getting-started/installation). Neptune logging won't be used")
                self.weblogger = False
        print(rs.fg)

