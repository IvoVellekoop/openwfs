---
title: PyDevice
layout: page
---


## Summary

<table>
<tr>
<td markdown="1">

**Summary:**

</td>
<td markdown="1">

Device adapter for control of Python objects in MicroManager

</td>
</tr>
<tr>
<td markdown="1">

**Author:**

</td>
<td markdown="1">

Ivo Vellekoop and Jeroen Doornbos

</td>
</tr>
<tr>
<td markdown="1">

**License:**

</td>
<td markdown="1">

BSD-3

</td>
</tr>
<tr>
<td markdown="1">

**Devices:**

</td>
<td markdown="1">

Stage, XYStage, Camera, Generic device

</td>
</tr>
</table>


## How it works

***This device is soon to be added***

The PyDevice runs its own Python instance, and exposes the objects to MicroManager.

This is done by recognising certain structures in the Python code, and transforming them to the required MicroManager objects.

Because exposing all your parameters and objects in Python to MicroManager might not be what you want, only the objects in a 'dict' called 'devices' will be imported.

Examples of these objects can be found in https://github.com/IvoVellekoop/openwfs/tree/master/openwfs/devices, which can also be installed & imported as the PyPI package OpenWFS.





## Rules for building devices:
**1.** Use the word 'devices' for your dict. 

**2.** Load only 1 file. So do not load multiple files containing their own devices dict. 
	You can write a Python script containing all your objects in a single devices dict and load that one.

**3.**
    *a)* Set the python home folder by the 'ModulePath' variable during initialisation. E.g. for my Anaconda install: C:\ProgramData\Anaconda3
	This path should contain the standard library and the site-packages (for packages).

   *b)* A Virtual environment is also seen as a separate interpreter. Make sure it contains the necessary packages to run the code.
   
   *c)* If 'ModulePath' was not set, it will find a python home folder itself. If the selected script was inside a virtual environment, it will select it.

**4.** Use only 1 interpreter. If you have loaded an interpreter in MicroManager this session and you want to use another interpreter: Restart the program.

**5.** *a)* Want a parameter in MicroManager? Expose it using the @property and @name.setter decorators. For example:

    @property
    def left(self) -> int:
        return self._top

    @left.setter
    def left(self, value: int):
        self._top = value

This will create a micromanager int property that you can see in the property manager. The -> int type hints are not required in python, 
		but are used by PyDevice. Don't forget them.


   *b)* Smarter type hints. You can also set ranges with the type hints. For example:

    @property
    def height(self) -> Annotated[int, {'min': 1, 'max': 960}]:
        return self._height

    @height.setter
    def height(self, value: int):
        self._height = value
        self._resized = True

   This creates a range with minima and maxima that you cannot cross. Useful to safeguard your hardware.


   *c)* astropy conventions. In order to get rid of unit errors in your devices, we sometimes require astropy units. For example:

    @property
    def measurement_time(self) -> Quantity[u.ms]:
        return self._measurement_time

    @measurement_time.setter
    def measurement_time(self, value):
        self._measurement_time = value.to(u.ms)

    This forces the user to use units, and is required by some of the device checks.

   *d)* `None`. MicroManager does not support `None` values, but they may be useful to indicate that a parameter is not set. As a workaround for floating point properties, a `None` in Python is converted to a `nan` in MicroManager, and vice-versa.

**6.** Specific devices need to follow specific structures. MicroManager has internal requirements of what a device should have to be such a device.
If that is a bit vague: a MicroManager stage needs properties like GetPositionUm, SetPositionUm etc. Otherwise, it is not a stage. 
Bootstrap.py checks your Python objects if they meet any of the requirements of any specific object, and marks them as such.

This allows you to build a script returning a numpy array, while MicroManager thinks it is a camera. 
Specific device requirements can be seen in bootstrap.py and in SPECIFIC_DEVICE_REQUIREMENTS.md.
