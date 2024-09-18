<script>
    import { unet, unet2 } from './dummy_layers';
    import { current_node_hash, proposed_node_hash, training_mode } from './stores';
    import Net from './Net.svelte';
    import Graph from './Graph.svelte';
    import CodeDiff from './CodeDiff.svelte';
    import LineChart from './LineChart.svelte';
    import Config from './Config.svelte';
    import Samples from './Samples.svelte';
    import '../app.css';

    
    let net = unet;


</script>

<!-- <LineChart />
<CodeDiff />
<Graph /> -->

<div class="main_container">
    <div class="unet">
        <Net />
    </div>
    <div class="performance" >
        <div class="loss_and_fid">
            <div class="loss">
                <LineChart />
            </div>
            <div class="fid">
            </div>
        </div>
        <div class="image_samples">
            <Samples />
        </div>
    </div>
    <div class="code_and_config">
        <div class="code">
            <CodeDiff />
        </div>
        <div class="config">
            <Config />
        </div>
    </div>
    <div class="graph">
        <Graph />
    </div>
</div>

<style>
    .main_container {
        height: 100vh;
        width: 100vw;   
        position: relative;
        background: linear-gradient(45deg, rgba(243, 236, 120, 0.3), rgba(175, 66, 97, 0.3));
        display: flex;
        justify-content: center;
        align-items: center;
    }
    /* level 1 */
    .unet {
        height: 30%;
        width: 100%;
        position: absolute;
        bottom: 20px;
        display: flex;
        flex-direction: column-reverse;
    }
    .performance, .code_and_config {
        height: 60%;
        width: 26%;
        position: absolute;
        right: 20px;
        top: 20px;
        background: white;
        display: flex;
        flex-direction: column;
        border: 4px solid black;
    }
    .code_and_config {
        left: 20px;
        width: 35%;
    }
    .graph {
    }
    /* level 2+ */
    /* performance */
    .loss_and_fid {
        flex: 1;
        display: flex;
        flex-direction: column;
        padding: 2px;
    }
    .loss {
        flex: 2;
    }
    .fid {
        flex: 1;
        display: none;
    }
    .image_samples {
        flex: 1;
    }
    /* code and config */
    .code {
        flex: 2;
        overflow: scroll;
    }
    .config {
        display: none;
        flex: 1;
    }

</style>