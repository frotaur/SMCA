# SMCA
LGCA-type cellular automaton, with addition of standard model inspired physics



# How to use ?
Use CreateConfig, you can modify the physics of the automaton with the basic building blocks. To create a new set of physics, simply open CreateConfig.py.

Inside the 'def make_config' function, there are several constants and dictionaries. These are the parameters of the cellular automaton. Here is what they mean :

`CONFIG_NAME` : name of the set of parameters. This set will be saved as a file named `CONFIG_NAME.json`

`WIDTH`, `HEIGHT` : size of the simulation window, in pixels.
`FPS` : maximum frames per second during simulation. Use if you want to slow down dynamics beyond what you are seeing.

`photon_interactions` dictionary : it governs which interactions produces photons.
    - `sticking_photon` : if True, sticking interaction generates photons
    - `protonaction_photon` : if True, proton action generates photons
    - `neutronaction_photon` : if True, neutron action generates photons

`photon_create_order` array : it governs the order of the photon generation. Just re-arrange the three strings inside (do NOT modify them) in the order you want the interactions to happen

`execution_order` array : it governs which actions, and in which order take place in the cellular automaton. Here you can include any number of strings, from a limited set. Adding a string more than once will mean that the action executes several times each time-step.
    - `count_particles` : activates the particle counting algorithm. Only useful for tracking, does not modify physics
    - `propagation_prot_neut_step` : propagates the protons and neutron one time-step
    - `propagation_photon_step` : propagates the photons one time-step
    - `scattering_step` : scatters protons and neutrons together
    - `protonaction_step`, `neutronaction_step` : makes the proton/neutron action execute
    - `absorption_step : makes the absorption of the photons
    - `sink_step` : activates the particle sink (i think)


`constants_dict` dictionary : contains the different constants governing the interactions. They are explained with comments in the code, and their type can be inferred from their value. TODO explain better.

`initialization_dict` dictionary : contains variable for initialization, for now not much.
    - `use` : if True, will use the random initialization. For now the False case has not been implemented, I think.
     - `proton_percent and neutron_percent` : values between 0. and 1., percentage-wise how many site should be occupied by (random direction) protons an nucleons. Actually the sum of those may go up to 7., because we can have up to 7 particles per site, but this will probably not be used.

That's it ! You don't need to worry about the rest of the code. If you run `python CreateConfig.py`, a file should be generated in `configurations\`, with the name you specified.

To run the program with this set of parameters, go to `main.py`. On top of the file, change the `CONFIG_NAME` to the appropriate one, and run the `python main.py`. It may take a while to compile and launch, but successive launches should be instant.

NOTE : if you want to iterate quickly on parameters, you can just put `None` in the CONFIG_NAME field of `main.py`. Then, launching the program will use the currently defined parameters in CreateConfig. That way, you do not need to run `python CreateConfig` each time you change something. If you want to access the current parameters, they should be save as `configuration/current.json`. 