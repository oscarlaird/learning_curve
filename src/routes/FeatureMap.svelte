<script>
    import { onMount } from "svelte";
    export let layer;
    const height_scale_factor = 10;
    const width_per_channel = 1;
    import { tweened } from "svelte/motion";
    import { fade } from "svelte/transition";
    let init_width = layer.channels * width_per_channel;
    let tweened_width = tweened(init_width, {duration: 400});
    $: tweened_width.set(layer.channels * width_per_channel);
</script>

<div class="feature_map_box" 
    style:height={layer.size * height_scale_factor}px
    style:width={$tweened_width}px
>
    <div class="channels_text">
        {layer.channels}
    </div>
    <div class="height_text">
        {layer.size} × {layer.size}
    </div>
</div>


<style>
    .feature_map_box {
        margin-top: 30px;
        border: 2px solid black;
        position: relative;
    }
    .channels_text {
        position: absolute;
        bottom: 100%;
        left: 0;
        width: 100%;
        display: flex;
        justify-content: center;
    }
    .height_text {
        position: absolute;
        left: 0;
        top: 50%;
        display: flex;
        justify-content: center;
        transform: translate(-50%, -50%) rotate(-90deg) translateY(50%);
        background-color: white;
        border: 1px solid black;
        outline: 1px dashed pink;
        padding: 0 10px;
        text-wrap: nowrap;
        font-size: 1rem;
    }
</style>

