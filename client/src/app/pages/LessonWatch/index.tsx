/**
 *
 * Watch Lesson
 *
 */
import * as React from 'react';
import styled from 'styled-components/macro';
import { StyleConstants } from '../../../styles/StyleConstants';
import { Helmet } from 'react-helmet-async';
import { NavBar } from '../../components/NavBar';
import {useQuery} from "../../../hooks/useQuery";

interface Props {}

export function LessonWatch(props: Props) {
  const query = useQuery();
  const imgPath = require("../../../../../lesson/" + query.get("lesson") + ".jpg");
  const videoPath = require("../../../../../lesson/" + query.get("lesson") + ".mp4");


  return (
    <>
      <Helmet>
        <title>{query.get('lesson')}</title>
        <meta
          name="description"
          content="The development team of this project"
        />
      </Helmet>
      <NavBar />
      <PageWrapperMain>
        <div className={'flex w-full bg-black justify-center'}>
            <div className={'flex flex-row justify-center w-full px-10'} style={{alignItems: 'center'}}>
                  <div className="mr-5" style={{width: '55%'}}>
                      <img className="img-height-fit mr-5" src={imgPath} />
                  </div>
                  <div className="mb-5" style={{width: '35%'}}>
                      <video style={{width: '100%'}} controls>
                        <source src={videoPath} type="video/mp4"/>
                    </video>
                  </div>
            </div>
        </div>
      </PageWrapperMain>
    </>
  );
}

export const PageWrapperMain = styled.div`
  display: flex;
  margin: 0 auto;
  box-sizing: content-box;
  height: calc(100vh - ${StyleConstants.NAV_BAR_HEIGHT});
`;
