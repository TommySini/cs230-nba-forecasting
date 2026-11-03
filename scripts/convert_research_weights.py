"""Expand MLX affine four-bit tensors without constructing a language model."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np
from safetensors import safe_open
from safetensors.numpy import save_file


def expand(packed, scales, biases):
    values = ((packed[:, :, None] >> np.arange(0,32,4,dtype=np.uint32)) & 15)
    values = values.reshape(packed.shape[0],-1,64).astype(np.float32)
    # One rounding to FP16 matches MLX's GPU dequantization. CPU MLX rounds
    # the multiply and addition separately and is deliberately not the reference.
    values *= scales.astype(np.float32)[:, :, None]
    values += biases.astype(np.float32)[:, :, None]
    return values.reshape(packed.shape[0],-1).astype(np.float16)


def convert(source, target, reference):
    source,target=Path(source),Path(target)
    config=json.loads((source/'config.json').read_text())
    if config['quantization']!={'group_size':64,'bits':4}:
        raise ValueError('Converter supports only the frozen affine four-bit/64 format')
    if target.exists():
        raise ValueError('Refusing to overwrite a conversion')
    files=[safe_open(str(p),framework='np') for p in sorted(source.glob('*.safetensors'))]
    keys={k:f for f in files for k in f.keys()}
    shard,weight_map={},{}
    shard_bytes,total_bytes,number,quantized=0,0,0,0
    with tempfile.TemporaryDirectory(prefix='nba-conversion-') as temporary:
        staging=Path(temporary)
        def flush():
            nonlocal number,shard,shard_bytes
            if not shard:return
            number+=1
            name=f'model-{number:05}.safetensors'
            save_file(shard,str(staging/name),metadata={'format':'pt'})
            weight_map.update({key:name for key in shard})
            print(json.dumps({'saved_shard':name,'bytes':shard_bytes,'tensors':len(shard)}),flush=True)
            shard,shard_bytes={},0
        for key in sorted(keys):
            if key.endswith(('.scales','.biases')):continue
            base=key[:-7] if key.endswith('.weight') else ''
            if base+'.scales' in keys:
                packed=keys[key].get_slice(key)
                shape=packed.get_shape()
                scales=keys[base+'.scales'].get_tensor(base+'.scales')
                biases=keys[base+'.biases'].get_tensor(base+'.biases')
                if scales.dtype!=np.float16 or biases.dtype!=np.float16:
                    raise ValueError('Unexpected scaling precision')
                value=np.empty((shape[0],shape[1]*8),dtype=np.float16)
                for start in range(0,shape[0],128):
                    value[start:start+128]=expand(packed[start:start+128],scales[start:start+128],biases[start:start+128])
                if hashlib.sha256(value[:1].tobytes()).hexdigest()!=reference[key]:
                    raise ValueError('Direct conversion differs from original MLX GPU sample: '+key)
                quantized+=1
            else:
                value=keys[key].get_tensor(key)
            if shard_bytes+value.nbytes>2*1024**3:flush()
            shard[key]=value
            shard_bytes+=value.nbytes
            total_bytes+=value.nbytes
        flush()
        if quantized!=len(reference):
            raise ValueError('Not every quantized matrix was checked')
        (staging/'model.safetensors.index.json').write_text(json.dumps({'metadata':{'total_size':total_bytes},'weight_map':weight_map},indent=2)+'\n')
        for path in source.iterdir():
            if path.is_file() and path.suffix in ('.json','.txt','.model') and path.name!='model.safetensors.index.json':
                shutil.copyfile(path,staging/path.name)
        config.pop('quantization',None)
        config.pop('quantization_config',None)
        config['torch_dtype']='float16'
        (staging/'config.json').write_text(json.dumps(config,indent=2)+'\n')
        target.mkdir()
        for path in sorted(staging.iterdir()):
            with path.open('rb') as src,(target/path.name).open('wb') as dest:
                shutil.copyfileobj(src,dest,length=16*1024**2)
        print(json.dumps({'quantized_matrices':quantized,'sampled_gpu_exact_matches':quantized,'total_bytes':total_bytes,'shards':number}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source');parser.add_argument('target');parser.add_argument('reference')
    args=parser.parse_args()
    convert(args.source,args.target,json.loads(Path(args.reference).read_text()))
