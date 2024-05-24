import { Box } from "@material-ui/core";
import { Tabs, Tab } from '@material-ui/core';
import {
  Streamlit,
  StreamlitComponentBase,
  withStreamlitConnection,
  ComponentProps
} from "streamlit-component-lib";
import React, { ReactNode } from "react";

interface State {
  activeStep: number;
}

const menu_options = ['Text Annotaion', 'Image Labeling' ,'In-Context Learning','Prompt Optimization'];

class DiscreteSlider extends StreamlitComponentBase<State> {
  public constructor(props: ComponentProps) {
    super(props);
    this.state = { activeStep: 0 };
  }

  public handleChange = (newValue: number) => {
    this.setState({ activeStep: newValue });
    Streamlit.setComponentValue(menu_options[newValue - 1]);  // Set the component value to the selected tab label
  };

  public render = (): ReactNode => {
    return (
      <div style={{ height: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
        <Box style={{ width: '37%', height: '60px', backgroundColor: '#f0f0f0', display: 'flex', alignItems: 'center', justifyContent: 'center', borderRadius: '8px', boxShadow: '0 4px 8px rgba(0, 0, 0, 0.1)' }}>
          <Tabs
            value={this.state.activeStep.toString()}
            onChange={(_, newValue) => this.handleChange(newValue)}
            aria-label="secondary tabs example"
            TabIndicatorProps={{ style: { backgroundColor: '#ff8c00' } }}
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
                      fontSize: '14px',
                      height: '60px',
                      padding: '0 20px',
                      textTransform: 'none'
                    }}
                  />
                );
              })
            }
          </Tabs>
        </Box>
      </div>
    );
  }
}

export default withStreamlitConnection(DiscreteSlider);
