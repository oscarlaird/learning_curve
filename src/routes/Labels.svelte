<!-- Labels.svelte -->
<script>
    export let width, height, margin, label, yoffset, xoffset;
    export let labelforx = false;
    export let labelfory = false;
  
    export let textanchor = 'end';
  </script>
  
  {#if width}
    <g class="labels-x-y text-sm">
      {#if labelforx}
        <text
          fill="white"
          x={+margin.left + width - 66 + xoffset}
          y={+height + margin.top - 40 + yoffset}
          text-anchor={textanchor}>{label}</text>
      {/if}
      {#if labelfory}
        <text
          fill="white"
          x={+margin.left + xoffset}
          y={-margin.top + 80 + yoffset}
          text-anchor={textanchor}>{label}</text>
      {/if}
    </g>
  {/if}