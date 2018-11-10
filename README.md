# WLDiscriminationNetwork
Assess different neural networks ability to discriminate between different signal known exactly cases

The idea is we will make simple stimuli using either ISETBio or ISETCam.  The stimuli will be known exactly.  We will then create sensor (or cone mosaic) images with photon noise only.  We will see how well the stimuli can be separated as we change the stimulus contrast.  We will make this assessment using the ideal Poisson discriminator, and then we will train various simple networks on the same task.  We will compare the network performance to the ideal Poisson discriminator.

This will be a tool to assess the computational burden and performance relative to ideal for various networks.  We will do this for popular networks and for networks that we think are simple and effective for this SKE task.
