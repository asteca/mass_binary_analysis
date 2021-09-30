
TODO:

- <s>Assign (m1, m2) masses, put both stars into the single systems' pool, and
   *then* estimate the single system's mass.</s>
- <s>Find the relation between G and the distance to the MS with secondary
  binary mass</s>
- <s>Split the code into binary probabilities assignment and mass estimation?</s>
- <s>Decide what to do with the "dip" in the G=[15, 17] range</s>
- <s>Define a binary ridge line using m1=m2 and assign binary probabilities 
  proportional to the distance to each ridge line/isochrone. This would remove
  the necessity for the `thresh_binar` parameter (which is rather arbitrary)</s>
- <s>Use *only* the envelope method to estimate binarity. Combined with 7.,
  this would take care of 4.</s>

- Use ASteCA's binary probability to identify binaries?
- Use LOWESS method to estimate the MSRL?
- Use RobustGP (https://github.com/syrte/robustgp) method to estimate the MSRL?
- Remove outliers (binaries mostly) when obtaining the envelope

- Estimate the MSRL (however) and use split the sequence in magnitude/mass
  bins. For each bin count the number of stars *below* the MSRL (Nb) and the
  average scatter of the MSRL in this region. Assume that this same number of
  stars and similar scatter must exist *above* the MSRL. For each mag/mass bin
  select randomly Nb stars from the scatter region *above* the MSRL, and
  assign these as single stars. The rest are thus binaries. Repeating this
  process allows us to estimate the mass fraction (q) as well as the fraction
  of binaries per mass interval.


## Articles with similar methods

* Modeling Unresolved Binaries of Open Clusters in the Color-Magnitude
  Diagram. I. Method and Application of NGC 3532, Li et al. (2020); Zotero
> Describes a method to estimate the parameters binary fraction and binary
> mass ratio
* [Bayesian Characterization of Main-sequence Binaries in the Old Open
  Cluster NGC 188, Cohen et al. (2019)](https://iopscience.iop.org/article/10.3847/1538-3881/ab59d7)
* [The Binary INformation from Open Clusters Using SEDs (BINOCS) Project:
  Reliable Photometric Mass Determinations of Binary Star Systems in Cluster,
  Thompson et al. (2021)](https://arxiv.org/abs/2101.07857)


## Description of the code

```
 run
    |--> readData
        |--> readINI
    |--> readData
        |--> loadClust
    |--> totalMass
       |--> IMF_CDF
    |--> binary
        |--> ID
            |--> clustHandle
                |--> splitEnv
                    |--> isochHandle
                        |--> interp
            |--> isochHandle
                |--> isochProcess
                    |--> readData
                        |--> loadIsoch
                    |--> move
            |--> clustHandle
                |--> split
            |--> clustHandle
                |--> singleMasses
    |--> binary
        |--> masses
            |--> isochHandle
                |--> isochProcess
                    |--> readData
                        |--> loadIsoch
                    |--> move
            |--> clustHandle
                |--> binarMasses
    |--> totMass
        |--> totalMass
            |--> get
                |--> getIMF
                |--> prepObsMass
                |--> tremmel
        |--> totalMass
            |--> get
                |--> getIMF
                |--> prepObsMass
                |--> tremmel
        |--> totalMass
            |--> extrapolate
```

### Generate the IMF's inverse CDF

Generate the inverse CDF for the selected IMF. This is done once at the beginning of the code, and allows the random sampling of the IMF later on.


### Split single/binary systems

Assign probabilities for each star of being a single or binary system. Two methods are currently supported:

* Envelope: 


