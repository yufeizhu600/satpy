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

import numpy as np

from satpy.composites import CompositeBase
from satpy.dataset import combine_metadata

LOG = logging.getLogger(__name__)

class MaxChlIndex(CompositeBase):
    """Maximum Chlorophyll Index compositor class.

    reference:
    https://github.com/senbox-org/optical-toolbox/tree/master/opttbx-s2msi-mci/src/main/java/eu/esa/opt/processor/mci.
    """

    def __call__(self, projectables, nonprojectables=None, **attrs):
        """"Generate the composite."""
        if len(projectables) != 3:
            raise ValueError(f"Expected 3 datasets, got {len(projectables)}")

        upper, peak, lower = self.match_data_arrays(projectables)
        info = combine_metadata(upper, peak, lower)
        info["name"] = self.attrs["name"]
        info.update(self.attrs)  # attrs from YAML/__init__
        info.update(attrs)  # overwriting of DataID properties

        # get wavelength and convert units to `nm`,
        # "where for OLCI, λ1, λ2 and λ3 are centered at 681, 708, and 753 nm respectively."
        upper_lambda, signal_lambda, lower_lambda = (
            pj.attrs["wavelength"].central * 1000 for pj in (upper, peak, lower)
        )

        # set numerator and denominator
        num = signal_lambda - lower_lambda
        denom = upper_lambda - lower_lambda

        # wavelength factor
        lambda_factor = num / denom

        mci = peak - lower - (upper - lower) * lambda_factor
        mci.attrs = info

        return mci


class MCISlope(CompositeBase):
    """Maximum Chlorophyll Index Slope compositor class.

    reference:
    https://github.com/senbox-org/optical-toolbox/tree/master/opttbx-s2msi-mci/src/main/java/eu/esa/opt/processor/mci.
    """

    def __call__(self, projectables, nonprojectables=None, **attrs):
        """"Generate the composite."""
        if len(projectables) != 3:
            raise ValueError(f"Expected 3 datasets, got {len(projectables)}")

        upper, peak, lower = self.match_data_arrays(projectables)
        info = combine_metadata(upper, peak, lower)
        info["name"] = self.attrs["name"]
        info.update(self.attrs)  # attrs from YAML/__init__
        info.update(attrs)  # overwriting of DataID properties

        # get wavelength and convert units to `nm`,
        # "where for OLCI, λ1, λ2 and λ3 are centered at 681, 708, and 753 nm respectively."
        upper_lambda, signal_lambda, lower_lambda = (
            pj.attrs["wavelength"].central * 1000 for pj in (upper, peak, lower)
        )

        # set denominator
        denom = upper_lambda - lower_lambda

        mci_slope = (upper - lower) / denom
        mci_slope.attrs = info

        return mci_slope


class MCIDerivedChla(CompositeBase):
    """Maximum Chlorophyll Index derived Chlorophyll compositor class.

    reference:
    https://github.com/senbox-org/optical-toolbox/tree/master/opttbx-s2msi-mci/src/main/java/eu/esa/opt/processor/mci.

    0 <= Chlorophyll <= 200 and where MCI > 0 and MCI_slope > -0.15
    """

    def __call__(self, projectables, nonprojectables=None, **attrs):
        """"Generate the composite."""
        if len(projectables) != 3:
            raise ValueError(f"Expected 3 datasets, got {len(projectables)}")

        upper, peak, lower = self.match_data_arrays(projectables)
        info = combine_metadata(upper, peak, lower)
        info["name"] = self.attrs["name"]
        info.update(self.attrs)  # attrs from YAML/__init__
        info.update(attrs)  # overwriting of DataID properties

        # get wavelength and convert units to `nm`,
        # "where for OLCI, λ1, λ2 and λ3 are centered at 681, 708, and 753 nm respectively."
        upper_lambda, signal_lambda, lower_lambda = (
            pj.attrs["wavelength"].central * 1000 for pj in (upper, peak, lower)
        )

        # set numerator and denominator
        num = signal_lambda - lower_lambda
        denom = upper_lambda - lower_lambda

        # wavelength factor
        lambda_factor = num / denom

        mci = peak - lower - (upper - lower) * lambda_factor

        mci_slope = (upper - lower) / denom

        chla = 6.166 * mci + 6.347

        # clip chla to a range
        chla = chla.clip(0, 200)

        # mask with MCI and MCI_slope
        chla = chla.where(np.any(mci > 0) and np.any(mci_slope > -0.15))

        chla.attrs = info

        return chla


class NormalizedDifferenceWaterIndex(CompositeBase):
    """Normalized Difference Water Index compositor class."""

    def __call__(self, projectables, nonprojectables=None, **attrs):
        """"Generate the composite."""
        if len(projectables) != 2:
            raise ValueError(f"Expected 2 datasets, got {len(projectables)}")

        projectables = self.match_data_arrays(projectables)
        info = combine_metadata(*projectables)
        info["name"] = self.attrs["name"]
        info.update(self.attrs)  # attrs from YAML/__init__
        info.update(attrs)  # overwriting of DataID properties

        ndwi = (projectables[0] - projectables[1]) / (projectables[0] + projectables[1])
        ndwi.attrs = info

        return ndwi
