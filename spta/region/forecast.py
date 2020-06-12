import numpy as np
from spta.region.function import FunctionRegionSeries


class ForecastModelRegion(FunctionRegionSeries):
    '''
    A FunctionRegion that assumes that a trained forecasting model is present at each point,
    so that a forecast is produced when the function is applied over a region.

    This means that applying this FunctionRegion to a region with another region,
    the result will be a spatio-temporal region, where each point (i, j) is a forecast series of
    the model at (i, j).

    Subclasses must provide a specific model function call.
    When creating an instance of of this class using a TrainingFunctionRegion, make sure
    that function region has dtype=object (to store a model object instead of a value).
    '''

    def function_at(self, point):
        '''
        Override the method that returns the function at each point (i, j). This is needed because
        this function region does not really store a function, it stores a model object.

        We want to have the model object to inspect some properties, the model is still
        retrievable using value_at(point).
        '''
        # the region stores the models at (i, j), extract it and return the forecast result.
        model_at_point = self.value_at(point)

        # a generic forecsat function, which will produce a NaN series when the model is None,
        # or the model forecast which must be specified by subclasses.
        def do_forecast_from_model(value_at_point):

            if model_at_point is None:
                # no model, return array of NaNs
                # note that we require forecast_len, which is provided only when the function is
                # applied! Hence the decoration of apply_to() method below.
                return np.repeat(np.nan, repeats=self.forecast_len)
            else:
                return self.forecast_from_model(model_at_point, self.forecast_len, value_at_point,
                                                point)

        return do_forecast_from_model

    def forecast_from_model(self, model_at_point, forecast_len, value_at_point, point):
        '''
        Subclasses must implement the specific call on the model that produces a forecast.
        '''
        raise NotImplementedError

    def apply_to(self, spt_region, output_len):
        '''
        Decorate to get the forecast series length from output_len, used by function_at.
        '''
        self.forecast_len = output_len
        return super(ForecastModelRegion, self).apply_to(spt_region, output_len)
