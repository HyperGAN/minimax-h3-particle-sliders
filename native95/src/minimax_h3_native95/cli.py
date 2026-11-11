"""Explicit local commands; check/help do not import numerical frameworks."""
import argparse,json
from pathlib import Path

def main(argv=None):
    parser=argparse.ArgumentParser(description='New native H3 full AV PG095 RAW cohort; native qualification pending')
    sub=parser.add_subparsers(dest='command',required=True);sub.add_parser('check')
    cache=sub.add_parser('cache')
    for key in('snapshot','pairs','output'):cache.add_argument('--'+key,required=True)
    cache.add_argument('--device',default='cpu');cache.add_argument('--duration',type=int,default=5)
    cache.add_argument('--resolution',choices=('480p','768p'),default='480p');cache.add_argument('--seed',type=int,default=7)
    train=sub.add_parser('train')
    for key in('config','output'):train.add_argument('--'+key,required=True)
    train.add_argument('--resume');train.add_argument('--stop-after',type=int)
    render=sub.add_parser('render-pair')
    for key in('artifact','snapshot','output','prompt'):render.add_argument('--'+key,required=True)
    render.add_argument('--device',default='cpu');render.add_argument('--seed',type=int,default=7);render.add_argument('--strength',type=float,default=1.)
    args=parser.parse_args(argv)
    if args.command=='check':
        from .contracts import BACKEND_NAME,CORE_REVISION,SOURCE_PROFILE,INITIALIZATION_POLICY,geometry
        print(json.dumps(dict(backend=BACKEND_NAME,core=CORE_REVISION,PG='0.9.5',source_profile=SOURCE_PROFILE,
            initialization=INITIALIZATION_POLICY,response_shapes=[geometry({'resolution':r})['response_shape']for r in('480p','768p')],
            selection='raw',ema_decay=None,bank=None,qualification='UNRUN'),indent=2));return
    if args.command=='cache':
        from .contracts import validate_installed_core
        validate_installed_core()
        from .native import PikaH3FlexBackend
        from .cache import prepare_cache
        runtime=PikaH3FlexBackend(args.snapshot,device=args.device)
        try:print(json.dumps(prepare_cache(runtime,json.loads(Path(args.pairs).read_text()),args.output,duration=args.duration,resolution=args.resolution,seed=args.seed),indent=2))
        finally:runtime.close()
    elif args.command=='train':
        from particle_sliders.lifecycle import SessionConfig
        from .lifecycle import open_training
        output=Path(args.output);output.mkdir(parents=True,exist_ok=False)
        config=SessionConfig.from_dict(json.loads(Path(args.config).read_text()))
        with open_training(config,resume=args.resume)as session:
            def saved(metrics):
                step=metrics['step'];full=output/f'FULL-{step:06d}.pt';raw=output/f'RAW-{step:06d}.pt'
                session.checkpoint(full);session.export_inference(raw,selection='raw')
                from .contracts import record
                from .journal import write_json
                commit=dict(schema='h3-native95-FULL-RAW-commit-v1',completed_steps=step,
                    config=config.as_dict(),FULL=record(full),RAW=record(raw),selection='raw',ema_decay=None)
                commit_path=output/f'COMMIT-{step:06d}.json';write_json(commit_path,commit)
                write_json(output/'LATEST.json',dict(commit=record(commit_path),completed_steps=step),replace=True)
            session.train(stop_after=args.stop_after,on_step=saved)
    else:
        from .render import render_pair
        print(json.dumps(render_pair(**{k:v for k,v in vars(args).items()if k!='command'}),indent=2))

if __name__=='__main__':main()
