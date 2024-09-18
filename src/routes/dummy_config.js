let config1 = `
# augmentation
- vertical flip
# Provide timestep information to model
- use timestep embedding
- embedding size: 64
`

let config2 = `
# augmentation
- horizontal flip
# Provide timestep information to model
- use timestep embedding
- embedding size: 64
`

config1 = config1.trim()
config2 = config2.trim()

export { config1, config2 }