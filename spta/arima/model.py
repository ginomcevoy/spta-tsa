from spta.model.base import ModelRegion


class ModelRegionArima(ModelRegion):
    '''
    A FunctionRegion that creates a forecast region using ARIMA models.
    See spta.model.base.ModelRegion for more details.

    TODO: support tp/tf?
    '''

    def forecast_from_model(self, model_at_point, forecast_len, value_at_point, point):
        '''
        Creates a forecast from a trained ARIMA model.

        When using statsmodels.tsa.arima_model.ARIMA:
            return model_at_point.forecast(forecast_len)[0]

        When using statsmodels.tsa.arima.model.ARIMA:
            return model_at_point.forecast(forecast_len)

        When using pmdarima.arima.ARIMA:
            return model_at_point.predict(forecast_len)
        '''
        return model_at_point.forecast(forecast_len)
