# marginal-flows
Prototyping a technique to use normalizing flows to learn the marginal prior distribution over some set of Holodeck parameters of interest, while training on simulated data from a much larger parameter space. Much of this latter space is latent/"nuisance" variables, hence the benefit of marginalization.

The basic structure of the normalizing flow training code draws on Nima Laal's [Pandora](https://github.com/NimaLaal/pandora). Reformatting/additions and the implementation of marginalization over some parameter subset are new.
