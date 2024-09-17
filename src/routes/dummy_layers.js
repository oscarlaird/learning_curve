// example of a unet specification

const unet = [
    {type: 'feature_map', size: 32, channels: 3, id: 1},
    {type: 'module', kernel: 3, title: 'Conv2D', id: 2},
    {type: 'feature_map', size: 32, channels: 64, id: 3},
    {type: 'module', title: 'Maxpool', id: 4},
    {type: 'feature_map', size: 16, channels: 64, id: 5},
    {type: 'module', kernel: 3, title: 'Conv2D', id: 6},
    {type: 'feature_map', size: 16, channels: 128, id: 7},
    {type: 'module', title: 'Maxpool', id: 8},
    {type: 'feature_map', size: 8, channels: 128, id: 9},
    // bottleneck
    {type: 'module', kernel: 3, title: 'Conv2D', id: 10},
    //
    {type: 'feature_map', size: 8, channels: 128, id: 11},
    {type: 'module', title: 'Upsample', id: 12},
    {type: 'feature_map', size: 16, channels: 128, id: 13},
    {type: 'module', kernel: 3, title: 'Conv2D', id: 14},
    {type: 'feature_map', size: 16, channels: 64, id: 15},
    {type: 'module', title: 'Upsample', id: 16},
    {type: 'feature_map', size: 32, channels: 64, id: 17},
    {type: 'module', kernel: 3, title: 'Conv2D', id: 18},
    {type: 'feature_map', size: 32, channels: 3, id: 19},

]
let unet2 = JSON.parse(JSON.stringify(unet))
unet2.splice(9, 2)
unet2[2] = {...unet2[2], channels: 128}

export { unet, unet2 }