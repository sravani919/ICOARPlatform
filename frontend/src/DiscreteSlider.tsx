import { Paper, Slider, styled, Button, StepContent, Stepper, Box, Step, StepLabel, Typography } from "@material-ui/core"
import {
  Streamlit,
  StreamlitComponentBase,
  withStreamlitConnection,
  ComponentProps
} from "streamlit-component-lib"
import React, { ReactNode, useState } from "react"

interface State {
  /**
   * The value specified by the user via the UI. If the user didn't touch this
   * widget's UI, the default value is used.
   */
  activeStep: number
}

const steps = [
  {
    label: 'Classification',
    description: ``,
  },
  {
    label: 'Visualisation',
    description:
      '',
  }
];

class DiscreteSlider extends StreamlitComponentBase<State> {
  public constructor(props: ComponentProps) {
    super(props)
    this.state = { activeStep: 0}
  }

  public handleNext = () => {
    console.log('Inside handle next - ', this.state.activeStep)
    const activeStep = this.state.activeStep
    this.setState({ activeStep: this.state.activeStep + 1 })
    Streamlit.setComponentValue(activeStep+1)
  };

  public handleBack = () => {
    const activeStep = this.state.activeStep
    this.setState({ activeStep: this.state.activeStep - 1 })
    Streamlit.setComponentValue(activeStep-1)
  };

  public handleReset = () => {
    this.setState({ activeStep: 0 })
    Streamlit.setComponentValue(0)
  };

  public render = (): ReactNode => {
    const vMargin = 7
    const hMargin = 20


    const options = this.props.args["options"]

    return (
      <Box style={{ width: '100%' }}>
      <Stepper activeStep={this.state.activeStep} >
        {steps.map((step, index) => (
          <Step key={step.label}>
            <StepLabel
              optional={
                index === 2 ? (
                  <Typography variant="caption">Last step</Typography>
                ) : null
              }
            >
              {step.label}
            </StepLabel>
          </Step>
        ))}
      </Stepper>
      <Box style={{ display: 'flex', flexDirection: 'row' }}>
        <Button
          color="inherit"
          disabled={this.state.activeStep === 0}
          onClick={this.handleBack}
          style={{  }}
        >
          Back
        </Button>
        <Box style={{ flex: '1 1 auto' }} />
        <Button onClick={this.handleNext} style={{  }} color='primary'>
          Next
        </Button>
      </Box>
    </Box>
    )
  }
}

export default withStreamlitConnection(DiscreteSlider)
