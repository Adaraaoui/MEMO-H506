import radiopyo as rp

# Simulation handler
# rp.LABARBE_CONFIG_FILE contains environment and beam definitions
sim = rp.UnitCell.from_toml(rp.LABARBE_CONFIG_FILE)
res = sim.run(t_span=[1e-9, 10]) #Run simulation from 1e-9 s to 10 s.

# Extract results to a nice pandas DataFrame.
df = res.to_pandas()

# ---------- Same but change beam definition ----------
peak_dose_rate  = 1000 # Gy/s (dose rate during ON time)
max_dose = 10 # Gy
period = 10e-3 # seconds
on_time = 1e-3 # seconds (=> Duty cycle of 10%)
beam = rp.PulsedBeam.from_peak_dose_rate(peak_dose_rate, period, on_time, max_dose)

sim = rp.UnitCell.from_toml(rp.LABARBE_CONFIG_FILE)
sim.set_beam(beam)

# Better to adapt simulation settings during ON and OFF time.
res = sim.prepare_chunked_run(t_span=[1e-9, 10],
                              atol=1e-4, #Absolute tolerance of the ODE Solver
                              rtol=1e-6, #Relative tolerance of the ODE Solver
                              max_step_size_on=1e-8,  #seconds
                              max_step_size_off=0.01, #seconds
                              ).run()
# /!\ For atol and rtol, the base unit for the ODE solver is [µmol/l]

df = res.to_pandas() # pandas.DataFrame

# Time integrate species concentrations.
s = res.integrate_species() # pandas.Series