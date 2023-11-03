import {
  Streamlit,
  StreamlitComponentBase,
  withStreamlitConnection,
  ComponentProps
} from "streamlit-component-lib"
import React, { ReactNode } from "react"
import {
  MDBCarousel,
  MDBCarouselItem,
} from 'mdb-react-ui-kit';
import dc_image from './assets/data-collection.jpeg'
import mm_image from './assets/analytics.jpeg'
import preprocessing_image from './assets/pre-processing.jpeg'
import "react-responsive-carousel/lib/styles/carousel.min.css"; // requires a loader
import { Carousel } from 'react-responsive-carousel';

class Corousel extends StreamlitComponentBase {
  public constructor(props: ComponentProps) {
    super(props)
    Streamlit.setComponentValue(0)
  }
  public render = (): ReactNode => {

    return (
      <Carousel autoPlay infiniteLoop>
      <div>
          <img src={dc_image} />
          <p style={{backgroundColor: 'grey'}}>Data Collection</p>
      </div>
      <div>
          <img src={preprocessing_image} />
          <p style={{backgroundColor: 'grey'}}>Pre-processing</p>
      </div>
      <div>
          <img src={mm_image} />
          <p style={{backgroundColor: 'grey'}}>Data Analysis</p>
      </div>
  </Carousel>
    // <MDBCarousel showControls showIndicators fade onClick={() => Streamlit.setComponentValue(0)}>
    //   <MDBCarouselItem
    //     className='w-100 d-block'
    //     itemId={1}
    //     src={dc_image}
    //     alt='...'
    //   >
    //     <h5>First</h5>
    //     <p>Nulla </p>
    //   </MDBCarouselItem>
    //   <MDBCarouselItem
    //     className='w-100 d-block'
    //     itemId={2}
    //     src={mm_image}
    //     alt='...'
    //   >
    //     <h5>Second slide label</h5>
    //     <p>Lorem ipsum dolor sit amet, consectetur adipiscing elit.</p>
    //   </MDBCarouselItem>

    //   <MDBCarouselItem
    //     className='w-100 d-block'
    //     itemId={3}
    //     src={dc_image}
    //     alt='...'
    //   >
    //     <h5>Third slide label</h5>
    //     <p>Praesent commodo cursus magna, vel scelerisque nisl consectetur.</p>
    //   </MDBCarouselItem>
    // </MDBCarousel>
    )
  }
}

export default withStreamlitConnection(Corousel)
