import { Box } from "@material-ui/core"
import {Tabs, Tab } from '@material-ui/core';
import {
  Streamlit,
  StreamlitComponentBase,
  withStreamlitConnection,
  ComponentProps
} from "streamlit-component-lib"
import React, { ReactNode } from "react"

interface State {
  /**
   * The value specified by the user via the UI. If the user didn't touch this
   * widget's UI, the default value is used.
   */
  activeStep: number
}

const menu_options = ["Data Collection", "Pre-processing", "Text Analysis", "Text Annotation", "Visualization", "Multi-media Analysis"]

class DiscreteSlider extends StreamlitComponentBase<State> {
  public constructor(props: ComponentProps) {
    super(props)
    this.state = { activeStep: 0}
  }
  public handleChange = (newValue: number) => {

    this.setState({ activeStep: newValue })
    Streamlit.setComponentValue(newValue)
  };

  public render = (): ReactNode => {

    return (
      <div style={{height: '100%'}}>
      <Box style={{ width: '100%',height: '80px', backgroundColor: 'black' }}>
      <div style={{}}>
      <div style={{width: '10%', display: 'inline-block', textAlign:'center', color: '#f56600', cursor: 'pointer', fontWeight: 'bold',
      fontSize: '20px'}}
      onClick={() => this.handleChange(0)}>
          ICOAR
        </div>
        <div style={{width: '80%', display: 'inline-block'}}>
          <Tabs
            value={this.state.activeStep.toString()}
            onChange={(_, newValue) => this.handleChange(newValue)}
            aria-label="secondary tabs example"
            style={{ color: 'white',height: '80px'}}
          >
            {
              menu_options.map((option, i) => {
                const key = i+1;
                console.log('key - ', key);
                return <Tab value={key.toString()} label={option} style={{height: '80px'}}/>
              })
            }
          </Tabs>
        </div>
        <div style={{width: '10%', display: 'inline-block', textAlign:'center', color: '#bbbbbb', cursor: 'pointer'}}
      onClick={() => {document.cookie="some_cookie_name=;"; this.handleChange(8)}}>
          Logout
        </div>
      </div>
    </Box>
    </div>

    )
  }
}

export default withStreamlitConnection(DiscreteSlider)
