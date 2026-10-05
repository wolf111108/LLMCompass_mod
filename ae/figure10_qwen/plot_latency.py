"""Plot the absolute Figure-10-scope request latency heatmap."""
import argparse
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('report',type=Path)
    args=p.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    rows=json.loads(args.report.read_text())['requests']
    inputs=sorted({r['input_length'] for r in rows})
    outputs=sorted({r['output_length'] for r in rows})
    values={(r['input_length'],r['output_length']):r['e2e_ms'] for r in rows}
    z=np.array([[values.get((s,g),float('nan')) for g in outputs] for s in inputs])
    fig,ax=plt.subplots(figsize=(max(5,len(outputs)*1.2),max(3,len(inputs)*.8)))
    im=ax.imshow(z,aspect='auto',cmap='viridis')
    ax.set_xticks(range(len(outputs)),outputs); ax.set_yticks(range(len(inputs)),inputs)
    ax.set_xlabel('Output tokens'); ax.set_ylabel('Input tokens')
    ax.set_title('Qwen / CIM: Figure-10-scope E2E estimate')
    fig.colorbar(im,ax=ax,label='Latency (ms)')
    for i in range(len(inputs)):
        for j in range(len(outputs)):
            ax.text(j,i,f'{z[i,j]:.1f}',ha='center',va='center',color='white')
    fig.tight_layout()
    fig.savefig(args.report.with_name('latency.png'),dpi=180)
    plt.close(fig)


if __name__=='__main__':
    main()
