

.PHONY : run nbhtml nbpdf x3d

ipynb : nb_proc/PiSox.ipynb nb_proc/PiSoXTolerances.ipynb nb_proc/PiSox_trades.ipynb nb_proc/ExplainBending.ipynb

nb_proc/%.ipynb : notebooks/%.ipynb
	mkdir -p nb_proc
	# Set kernel name to how the kernel is called when inside the environment
	# Notebook meta data may have global kernel name
	# Note output dir is relative to input dir
	# so we build in same dir and use ../$@
	cd notebooks && jupyter nbconvert --ExecutePreprocessor.kernel_name=python3 --ExecutePreprocessor.timeout=1800 --to notebook --execute $(notdir $<) --output ../$@
