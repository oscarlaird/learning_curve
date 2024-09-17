<script>
    let viewer_dist = 100;
    let rotX = 0;
    let rotY = 0;
    let rotZ = 0;
</script>

<input type="range" bind:value={viewer_dist} min="-1000" max="1000" step="1" /> {viewer_dist}
<input type="range" bind:value={rotX} min="-180" max="180" step="1" /> {rotX}
<input type="range" bind:value={rotY} min="-180" max="180" step="1" /> {rotY}
<input type="range" bind:value={rotZ} min="-180" max="180" step="1" /> {rotZ}

<div class="scene">
    <div class="cube"
        style:transform={`translateZ(${viewer_dist}px) rotateX(${rotX}deg) rotateY(${rotY}deg) rotateZ(${rotZ}deg)`}>
      <div class="face front"></div>
      <div class="face back"></div>
      <div class="face left"></div>
      <div class="face right"></div>
      <div class="face top"></div>
      <div class="face bottom"></div>
    </div>
  </div>
  


<style>
.scene {
  border: 1px solid black;
  background-color: red;
  perspective: 800px;
  width: 200px;
  height: 200px;
  position: relative;
  margin: 100px auto;
  transform: translateZ(0px);
}

/* Cube settings */
.cube {
  width: 100px;
  height: 100px;
  position: relative;
  transform-style: preserve-3d;
  /* transform: translateZ(-900px) rotateX(-30deg) rotateY(-30deg); */
  /* animation: rotate 10s infinite linear; */
}

.face {
  position: absolute;
  width: 100px;
  height: 100px;
  background-color: rgba(0, 128, 255, 0.95);
  border: 2px solid black;
}

/* Position each face of the cube */
.front  { transform: translateZ(50px); }
.back   { transform: rotateY(180deg) translateZ(50px); background-color: rgba(255, 0, 0, 0.99); }
.left   { transform: rotateY(-90deg) translateZ(50px); }
.right  { transform: rotateY(90deg) translateZ(50px); }
.top    { transform: rotateX(90deg) translateZ(50px); }
.bottom { transform: rotateX(-90deg) translateZ(50px); }

/* Optional rotation animation */
@keyframes rotate {
  from {
    transform: rotateX(-30deg) rotateY(-30deg);
  }
  to {
    transform: rotateX(-30deg) rotateY(-390deg);
  }
}

</style>