<script>
    import { unet, unet2 } from './dummy_layers';
    import Net from './Net.svelte';
    
    let net = unet;


</script>

<button on:click={() => net = (net === unet) ? unet2 : unet}>
    Toggle UNet
</button>

<Net layers={net} />
