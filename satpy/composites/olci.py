#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2023 Satpy developers
#
# This file is part of satpy.
#
# satpy is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# satpy is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE.  See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with
# satpy.  If not, see <http://www.gnu.org/licenses/>.
"""Composite classes for the Sentinel-3 OLCI instrument."""

import logging

from satpy.composites import CompositorBase

LOG = logging.getLogger(__name__)


class MCIDerivedChla(CompositorBase):
    """MCI compositor 
    (MCI <= 0 or  MCI_slope > -0.15) and (Chlorophyll >= 0 and Chlorophyll <= 200)      
    """

    def __call__(self, projectables, nonprojectables=None, **attrs):
        if len(projectables) != 3:
            raise ValueError(f"Expected 3 datasets, got {len(projectables)}")

        projectables = self.match_data_arrays(projectables)
        info = combine_metadata(*projectables)
        info['name'] = self.attrs['name']
        info.update(self.attrs)  # attrs from YAML/__init__
        info.update(attrs)  # overwriting of DataID properties

        proj.attrs = info
        # return proj

        lambda_1 = projectables[0].attrs["wavelength"].central * 1000
        lambda_2 = projectables[1].attrs["wavelength"].central * 1000
        lambda_3 = projectables[2].attrs["wavelength"].central * 1000

        L1, L2, L3  = self.match_data_arrays(projectables)
        
        MCI = L2 - L1 - (lambda_2 - lambda_1) / (lambda_3 - lambda_1) * (L3 -L1)

        MCI_slope = (L3 - L1) / (lambda_3 - lambda_1)

        Chla = 6.166 * MCI + 6.347

        Chla = da.clip(Chla, 0, 200)

        Chla = Chla.where(MCI <= 0)

        Chla = chla.where(MCI_slope > -0.15)

        Chla.attrs = info

        return Chla



class SimulatedGreen(GenericCompositor):
    """A single-band dataset resembling a Green (0.55 µm) band.

    This compositor creates a single band product by combining three
    other bands in various amounts. The general formula with
    dependencies (d) and fractions (f) is::

        result = d1 * f1 + d2 * f2 + d3 * f3

    See the `fractions` keyword argument for more information.
    Common used fractions for ABI data with C01, C02, and C03 inputs include:

    - SatPy default (historical): (0.465, 0.465, 0.07)
    - `CIMSS (Kaba) <https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2018EA000379>`_: (0.45, 0.45, 0.10)
    - `EDC <http://edc.occ-data.org/goes16/python/>`_: (0.45706946, 0.48358168, 0.06038137)

    """

    def __init__(self, name, fractions=(0.465, 0.465, 0.07), **kwargs):
        """Initialize fractions for input channels.

        Args:
            name (str): Name of this composite
            fractions (iterable): Fractions of each input band to include in the result.

        """
        self.fractions = fractions
        super(SimulatedGreen, self).__init__(name, **kwargs)

    def __call__(self, projectables, optional_datasets=None, **attrs):
        """Generate the single band composite."""
        c01, c02, c03 = self.match_data_arrays(projectables)
        res = c01 * self.fractions[0] + c02 * self.fractions[1] + c03 * self.fractions[2]
        res.attrs = c03.attrs.copy()
        return super(SimulatedGreen, self).__call__((res,), **attrs)

