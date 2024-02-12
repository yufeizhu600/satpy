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
import numpy as np
import logging

from satpy.composites import CompositeBase
from satpy.dataset import combine_metadata
import dask.array as da

LOG = logging.getLogger(__name__)


class MCIDerivedChla(CompositeBase):
    """Maximum Chlorophyll Index derived Chlorophyll compositor.
    MCI and MCI_slope reference:

    https://github.com/senbox-org/optical-toolbox/tree/master/opttbx-s2msi-mci/src/main/java/eu/esa/opt/processor/mci
    (MCI <= 0 or  MCI_slope > -0.15) and (Chlorophyll >= 0 and Chlorophyll <= 200)
    """

    def __call__(self, projectables, nonprojectables=None, **attrs):
        if len(projectables) != 3:
            raise ValueError(f"Expected 3 datasets, got {len(projectables)}")

        projectables = self.match_data_arrays(projectables)
        info = combine_metadata(*projectables)
        info["name"] = self.attrs["name"]
        info.update(self.attrs)  # attrs from YAML/__init__
        info.update(attrs)  # overwriting of DataID properties

        # get wavelength and convert units, "where for OLCI, λ1, λ2 and λ3 are centered at 681, 708, and 753 nm respectively."
        upper_lambda = np.floor(projectables[0].attrs["wavelength"].central * 1000)
        signal_lambda = np.floor(projectables[1].attrs["wavelength"].central * 1000)
        lower_lambda = np.floor(projectables[2].attrs["wavelength"].central * 1000)

        # set numerator and denominator
        num = signal_lambda - lower_lambda
        denom = upper_lambda - lower_lambda

        # inverse wavelength delta needed for baseline slope calculation
        inverse_delta = 1.0 / denom

        # wavelength factor
        lambda_factor = num / denom

        upper, peak, lower = self.match_data_arrays(projectables)

        mci = peak - lower - (upper - lower) * lambda_factor

        mci_slope = (upper - lower) * inverse_delta

        chla = (6.166 * mci + 6.347).clip(0, 200)

        # chla = 6.166 * mci + 6.347

        # chla = chla.clip(0, 200)

        chla.where(mci <= 0, np.NaN)

        chla.where(mci_slope <= -0.15, np.NaN)

        chla.attrs = info

        return chla
