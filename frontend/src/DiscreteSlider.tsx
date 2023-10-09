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
    description: `Classification`,
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
            <StepContent>
              <Typography>{step.description}</Typography>
              <Box style={{  }}>
                <div>
                  <Button
                    variant="contained"
                    onClick={this.handleNext}
                    style={{  }}
                  >
                    {index === steps.length - 1 ? 'Finish' : 'Continue'}
                  </Button>
                  <Button
                    disabled={index === 0}
                    onClick={this.handleBack}
                    style={{  }}
                  >
                    Back
                  </Button>
                </div>
              </Box>
            </StepContent>
          </Step>
        ))}
      </Stepper>
      {this.state.activeStep === steps.length && (
        <Paper square elevation={0} style={{  }}>
          <Typography>All steps completed - you&apos;re finished</Typography>
          <Button onClick={this.handleReset} style={{  }}>
            Reset
          </Button>
        </Paper>
      )}
    </Box>
    )
  }
}

export default withStreamlitConnection(DiscreteSlider)
