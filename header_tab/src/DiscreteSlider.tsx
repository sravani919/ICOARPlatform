import { Box } from "@material-ui/core";
import { Tabs, Tab } from '@material-ui/core';
import {
  Streamlit,
  StreamlitComponentBase,
  withStreamlitConnection,
  ComponentProps
} from "streamlit-component-lib";
import React, { ReactNode } from "react";

import icoarLogo from './icoar_logo.png';  // Ensure the logo image is in the same directory

interface State {
  activeStep: number;
}

const menu_options = ["Data Collection", "Pre-processing", "Text Analysis", "Visualization", "Multi-media Analysis", "AI-Assisted Features"];

class DiscreteSlider extends StreamlitComponentBase<State> {
  public constructor(props: ComponentProps) {
    super(props);
    this.state = { activeStep: 0 };
  }

  public handleChange = (newValue: number) => {
    this.setState({ activeStep: newValue });
    Streamlit.setComponentValue(newValue);
  };

  public render = (): ReactNode => {
    return (
      <div style={{ height: '100%' }}>
        <Box style={{ width: '100%', height: '60px', backgroundColor: '#f0f0f0', display: 'flex', alignItems: 'center', justifyContent: 'space-between', boxShadow: '0 4px 8px rgba(0, 0, 0, 0.1)' }}>
          <div style={{ display: 'flex', alignItems: 'center', width: '10%', cursor: 'pointer' }} onClick={() => this.handleChange(0)}>
            <img src={icoarLogo} alt="ICOAR Logo" style={{ height: '40px', marginRight: '8px' }} />
            <div style={{ color: '#ff8c00', fontWeight: 'bold', fontSize: '20px' }}>ICOAR</div>
          </div>
          <div style={{ width: '80%', display: 'flex', justifyContent: 'center' }}>
            <Tabs
              value={this.state.activeStep.toString()}
              onChange={(_, newValue) => this.handleChange(newValue)}
              aria-label="secondary tabs example"
              TabIndicatorProps={{ style: { backgroundColor: '#ff8c00' } }}
              style={{ width: '100%' }}
              variant="fullWidth"
            >
              {
                menu_options.map((option, i) => {
                  const key = i + 1;
                  return (
                    <Tab
                      key={key}
                      value={key.toString()}
                      label={option}
                      style={{
                        color: '#333',
                        fontWeight: 'bold',
                        fontSize: '16px',
                        height: '60px',
                        textTransform: 'uppercase'
                      }}
                    />
                  );
                })
              }
            </Tabs>
          </div>
          <div style={{ width: '10%', textAlign: 'center', color: '#bbbbbb', cursor: 'pointer' }}
            onClick={() => { document.cookie = "some_cookie_name=;"; this.handleChange(7); }}>
            Account
          </div>
        </Box>
      </div>
    );
  }
}

export default withStreamlitConnection(DiscreteSlider);
