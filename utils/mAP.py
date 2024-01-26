import cv2, glob
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px 

def get_mAP(mAP_file):
    with open(mAP_file, 'r') as f:
        lines = f.readlines()
        epoch = {}
        maps = []
        for i,line in enumerate(lines):
            epoch[i+1] = float(line.split(': ')[1].strip('\n'))*100
            maps.append(epoch[i+1])
        return epoch, maps


if __name__ == '__main__':

    dir = 'Output/Pytorch'
    paths = glob.glob(dir+'/*/*mAP*')

    mAPs, maps_all = {}, []
    for mAP_file in sorted(paths):
        if '64' in mAP_file: continue
        nImgs = int(mAP_file.split('/')[-2].split('-')[-1])
        mAPs[nImgs], maps = get_mAP(mAP_file)
        maps_all.append(maps)
        
    # Creating the Figure instance
    # fig = px.line(mAPs)
    fig = px.line(mAPs,
                  title='<b>mAP Comparison</b>',
                  )
    fig['layout']['xaxis']['title'] = 'Epoch Number'
    fig['layout']['yaxis']['title'] = 'mAP Score (%)'
    fig['layout']['legend']['title'] = 'nImgs'
    fig.update_traces(line_width=2)
    fig.update_layout(
                    font=dict(
                                family="Courier New, monospace, bold",
                                size=14,  # Set the font size here
                                color="RebeccaPurple",
                                
                            ),
                    )
    
    fig.write_image("mAPs.png")
    
    fig.update_traces(line_width=4)
    fig.update_layout(
                    title   = dict  (
                                    text        = '<b>mAP Comparison</b>',
                                    x           = 0.5,
                                    y           = 0.95,
                                    xanchor     = 'center',
                                    yanchor     = 'top',
                                    font        = dict  (
                                                        size    = 32,
                                                        color   = '#333333', 
                                                        family  = 'Avenir Light',
                                                        ),
                                    ),
                    font    = dict  (
                                    family      = "Courier New, monospace, bold",
                                    size        = 20,  # Set the font size here
                                    color       = "RebeccaPurple",
                                    ),
                    legend  = dict  (
                                    title       = '<b>Number of Images</b>',
                                    orientation = 'h',
                                    yanchor     = 'top',
                                    y           = 1.01,
                                    xanchor     = 'right',
                                    x           = 1       
                                    ),
                    xaxis   = dict  (
                                    title       = '<b>Epoch Number</b>',
                                    tickfont    = dict  (
                                                        size    = 20,
                                                        color   = '#333333', 
                                                        family  = 'Avenir Light',
                                                        ),
                                    ),
                    yaxis   = dict  (
                                    title       = '<b>mAP Score (%)</b>',
                                    tickfont    = dict  (
                                                        size    = 20,
                                                        color   = '#333333', 
                                                        family  = 'Avenir Light',
                                                        ),
                                    ),
                    width   = 1200, 
                    height  = 800,
                    # margin  = dict  (
                    #                 l=100,
                    #                 r=50,
                    #                 b=100,
                    #                 t=100,
                    #                 pad=4
                    #                 ),
                    )
    
    
    fig.write_image("mAPs_new.png")